import torch
from jaxtyping import Float

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: Float[torch.Tensor, "batch_size seq_len dim"]) -> Float[torch.Tensor, "batch_size seq_len 1"]:
        # (B, seq_len, dim) -> (B, seq_len, 1)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Float[torch.Tensor, "batch_size seq_len dim"]) -> Float[torch.Tensor, "batch_size seq_len dim"]:
        # dim : (B, seq_len, dim) -> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)