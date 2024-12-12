import torch
from typing import Tuple
from jaxtyping import Float, Int

OPS = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]]).reshape(1, 4, 1, 2)  # (1, 4, 1, 2)
# The reason we use 4 individual frames here is that we only need SO(3) invariance/equivariance since amino acids 
# are chiral molecules. Reflection of the structure will lead to a different structure. If O(3) invariance/equivariance
# is required, all 8 frames are necessary.


@torch.amp.autocast("cuda", enabled=False)
def create_frames(
    x: Float[torch.Tensor, "batch_size seq_len num_vectors 3"], 
    common_token_mask: Float[torch.Tensor, "batch_size seq_len"]) -> Tuple[
        Float[torch.Tensor, "batch_size 4 3 3"],
        Float[torch.Tensor, "batch_size 1 3"]] :
    """
    Create frames with Principal Component decomposition.
    Args:
        x: (B, S, N, 3)
        common_token_mask: (B, S)
    Returns:
        frames: (B, 4, 3, 3)
        center: (B, 1, 3)
    """
    # compute the center of the data
    batch_size, seq_len, num_nodes, _ = x.shape
    input_x = (x * common_token_mask.unsqueeze(-1).unsqueeze(-1)).reshape(batch_size, seq_len * num_nodes, 3)  # (B, S*N, 3)
    center = input_x.sum(dim=1, keepdim=True) / (
        num_nodes * common_token_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
    )  # (B, 1, 3)
    centered_x = ((x - center.unsqueeze(2)) * common_token_mask.unsqueeze(-1).unsqueeze(-1)).reshape(
        batch_size, seq_len * num_nodes, 3
    )  # (B, S*N, 3)
    # compute the covariance matrix
    # (B, 3, S*N) @ (B, S*N, 3) -> (B, 3, 3)
    cov = torch.einsum("bij,bjk->bik", centered_x.permute(0, 2, 1), centered_x)
    # compute the eigenvectors
    # (B, 3, 3) -> (B, 3, 2)
    eigenvectors = torch.linalg.eigh(cov, UPLO="U")[1][:, :, :2].to(cov.device)
    # compute the frames
    # (B, 1, 3, 2) x (1, 4, 1, 2) -> (B, 4, 3, 2)
    frames = eigenvectors.unsqueeze(1) * OPS.to(eigenvectors.device)
    # cat[(B, 4, 3, 2), (B, 4, 3, 1)] -> (B, 4, 3, 3)
    frames = torch.cat([frames, torch.linalg.cross(frames[..., 0], frames[..., 1]).unsqueeze(-1)], dim=-1)
    # apply the frames to the data
    # (B, 4, 3, 3) @ (B, S*N, 3) -> (B, 4, S*N, 3) -> (B*4, S, N*3)
    return frames.detach(), center.detach()

def apply_frames(
        x: Float[torch.Tensor, "batch_size seq_len num_vectors 3"], 
        common_token_mask: Float[torch.Tensor, "batch_size seq_len"],
        frames: Float[torch.Tensor, "batch_size 4 3 3"],
        center: Float[torch.Tensor, "batch_size 1 3"]) -> Float[torch.Tensor, "fbatch_size seq_len 3"]:
    """
    Apply the frames to the input.
    Args:
        x: (B, S, N, 3)
        common_token_mask: (B, S)
        frames: (B, 4, 3, 3)
        center: (B, 1, 3)
    Returns:
        output: (B*4, S, N*3)
    """
    batch_size, seq_len, num_nodes, _ = x.shape
    # (B, 4, 3, 3) @ (B, 4, S*N, 3) -> (B, 4, S*N, 3)
    centered_x = ((x - center.unsqueeze(2)) * common_token_mask.unsqueeze(-1).unsqueeze(-1)).reshape(
        batch_size, seq_len * num_nodes, 3
    )  # (B, S*N, 3)
    framed_x = torch.einsum("boij,bsj->bosi", frames.transpose(2, 3), centered_x).reshape(
        batch_size * 4, seq_len, num_nodes * 3
    )
    return framed_x

def invert_frames(
        framed_output: Float[torch.Tensor, "fbatch_size seq_len dim"],
        common_token_mask: Float[torch.Tensor, "batch_size seq_len"],
        frames: Float[torch.Tensor, "batch_size 4 3 3"], 
        center: Float[torch.Tensor, "batch_size 1 3"] = None) -> Float[torch.Tensor, "batch_size seq_len 3"]:
    """
    Invert the frames to get the output under original coordinates.
    Args:
        framed_output: (B*4, S, N*3)
        common_token_mask: (B, S)
        frames: (B, 4, 3, 3)
        center: (B, 1, 3)
    Returns:
        output: (B, S, N*3)
    """
    # (B, 4, 3, 3) @ (B, 4, S*N, 3) -> (B, 4, S*N, 3)
    output = torch.einsum("boij,bosj->bosi", frames, framed_output.reshape(frames.shape[0], 4, -1, 3)).mean(dim=1)
    if center is not None:
        output = output + center
    return output * common_token_mask.unsqueeze(-1)
