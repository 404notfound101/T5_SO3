import torch
from jaxtyping import Float, Int, Bool
from GraphT5_3D.nn.modules.RoPE import rotary_pos_emb

class MultiheadAttentionZero(torch.nn.Module):
    def __init__(self, dim: int, heads: int, attn_drop: float) -> None:
        super().__init__()
        self.heads = heads
        self.dim = dim // heads
        self.scale = self.dim**-0.5
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)
        self.attn_drop = attn_drop
        self.proj = torch.nn.Linear(dim, dim, bias=False)
        self.cached_k = None
        self.cached_v = None

    def reset_cache(self):
        self.cached_k = None
        self.cached_v = None

    def forward(
        self,
        q_x: Float[torch.Tensor, "batch_size target_seq_len dim"],
        kv_x: Float[torch.Tensor, "batch_size source_seq_len dim"],
        freq_q: Float[torch.Tensor, "target_seq_len head_dim//2"],
        freq_kv: Float[torch.Tensor, "source_seq_len head_dim//2"],
        key_padding_mask: Float[torch.Tensor, "batch_size source_seq_len"],
        is_causal: bool = False,
        cache_kv: bool = False,
    ) -> Float[torch.Tensor, "batch_size source_seq_len dim"]:
        """
        args:
            q_x: (B, T, D)
            kv_x: (B, S, D)
            freq_q: (T, D//2)
            freq_kv: (S, D//2)
            key_padding_mask: (B, S)
            is_causal: bool
            cache_kv: bool
        return:
            out: (B, T, D)
        """
        q = self.q(q_x).reshape(q_x.shape[0], q_x.shape[1], self.heads, self.dim).permute(0, 2, 1, 3)
        k = self.k(kv_x).reshape(kv_x.shape[0], kv_x.shape[1], self.heads, self.dim).permute(0, 2, 1, 3)
        v = self.v(kv_x).reshape(kv_x.shape[0], kv_x.shape[1], self.heads, self.dim).permute(0, 2, 1, 3)
        if self.cached_k is not None and cache_kv:
            k = torch.cat([self.cached_k, k], dim=2)
            v = torch.cat([self.cached_v, v], dim=2)
            self.cached_k = k
            self.cached_v = v

        q = rotary_pos_emb(q, freq_q)
        k = rotary_pos_emb(k, freq_kv)
        attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, -1, q_x.shape[1], -1).bool()
        out = self.proj(
            torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, self.attn_drop, is_causal=is_causal)
            .permute(0, 2, 1, 3)
            .reshape(q_x.shape)
        )

        return out
    
class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = 2 * dim

        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        swish = torch.nn.functional.silu(self.w1(x))  # Apply first transformation
        x_V = self.w3(x)
        x = swish * x_V  # Apply contraction to original dimension
        x = self.w2(self.dropout(x))  # Apply optional additional transformation
        return x