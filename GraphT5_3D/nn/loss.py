import torch
from typing import List
from jaxtyping import Float
from GraphT5_3D.data.tokenizer import ProteinTokenizer
from GraphT5_3D.nn.model import TransformerInput, DecoderOutput

class NextTokenLoss(torch.nn.Module):
    def __init__(self, config, tokenizer: ProteinTokenizer, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_los_weights = config.get("seq", 1.0)
        self.coord_loss_weights = config.get("coord", 1.0)

    def forward(self, output: DecoderOutput, target: TransformerInput, **kwargs) -> Float:
        """
        Due the the nature of decoder, left padding is used. Therefore, we need to ignore both cls and pad tokens, 
        in which case, we cannot rely on the default ignore_index in cross_entropy. Moreover, MSE loss also need 
        custom masking to reduce the loss to only the valid tokens.
        """
        # Mask cls and pad tokens, (B, S)
        loss_mask = torch.logical_and(target.sequence != self.tokenizer.pad, target.sequence != self.tokenizer.cls)
        seq_pred, coord_pred = output.invert_frames() # (B, S, V), (B, S, N, 3)
        seq_loss = torch.nn.functional.cross_entropy(seq_pred, target.sequence, reduction="none").sum(dim=-1) # (B, S)
        reduced_seq_loss = torch.sum(seq_loss * loss_mask) / torch.sum(loss_mask)
        loss_mask = loss_mask.unsqueeze(-1).unsqueeze(-1) # (B, S, 1, 1)
        coord_loss = torch.nn.functional.mse_loss(
            (coord_pred*loss_mask).reshape(-1, 3), 
            (target.coords*loss_mask).reshape(-1, 3), 
            reduction="none").sum(dim=-1) # (B*S, N)
        reduced_coord_loss = torch.sum(coord_loss * loss_mask.reshape(-1).unsqueeze(-1)) / (torch.sum(loss_mask) * 3)
        loss = self.seq_los_weights * reduced_seq_loss + self.coord_loss_weights * reduced_coord_loss
        return {"loss": loss, "seq_loss": seq_loss, "coord_loss": coord_loss}