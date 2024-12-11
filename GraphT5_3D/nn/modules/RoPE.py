import torch
from jaxtyping import Float
def precomputed_theta_pos_frequencies(head_dim: int, seq_len: int, theta: int = 10000) -> Float[torch.Tensor, "seq_len head_dim//2"]:
    """
    Args:
        head_dim: int
        seq_len: int
        theta: int
    Returns:
        freq: (S, D//2)
    """
    freq_numerator = torch.arange(0, head_dim, 2, dtype=torch.float32)
    theta = 1 / (theta ** (freq_numerator / head_dim))
    pos = torch.arange(0, seq_len, dtype=torch.float32)
    freq = torch.outer(pos, theta)
    return torch.polar(torch.ones_like(freq), freq)


@torch.amp.autocast("cuda", enabled=False)
def rotary_pos_emb(
    x: Float[torch.Tensor, "batch_size num_heads seq_len dim"], 
    freq: Float[torch.Tensor, "seq_len head_dim//2"]) -> Float[torch.Tensor, "batch_size num_heads seq_len dim"]:
    """
    Args:
        x: (B, H, S, D)
        freq: (S, D//2)
    Returns:
        x: (B, H, S, D)
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # (B, H, S, D//2)
    x_rot = x_complex * freq.unsqueeze(0).unsqueeze(0)  # (B, H, S, D//2)
    x_out = torch.view_as_real(x_rot).reshape(x.shape).to(x.dtype)
    return x_out