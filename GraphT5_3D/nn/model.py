import torch
from jaxtyping import Float, Int64
from dataclasses import dataclass
from GraphT5_3D.nn.modules.RoPE import precomputed_theta_pos_frequencies
from GraphT5_3D.nn.modules.norm import RMSNorm
from GraphT5_3D.nn.modules.transformer import MultiheadAttentionZero, FeedForward

@dataclass
class InFrameInput:
    EncoderSequence: Int64[torch.Tensor, "batch_size e_seq_len"]
    Encodercoords: Float[torch.Tensor, "batch_size e_seq_len num_atoms*3"]
    Encodermask: Float[torch.Tensor, "batch_size e_seq_len"]
    DecoderSequence: Int64[torch.Tensor, "batch_size d_seq_len"]
    Decodercoords: Float[torch.Tensor, "batch_size d_seq_len num_atoms*3"]
    Decodermask: Float[torch.Tensor, "batch_size d_seq_len"]

@dataclass
class InFrameOutput:
    logits: Float[torch.Tensor, "batch_size d_seq_len vocab_size"]
    coords_in_frame: Float[torch.Tensor, "batch_size d_seq_len num_atoms*3"]


class EncoderBlock(torch.nn.Module):
    """
    low-level implementation of Transformer encoder block with RoPE
    """

    def __init__(self, dim: int, heads: int, attn_drop: float, dense_drop: float) -> None:
        super().__init__()
        self.pre_attn_norm = RMSNorm(dim)
        self.attn = MultiheadAttentionZero(dim, heads, attn_drop=attn_drop)
        self.pre_ffw_norm = RMSNorm(dim)
        self.ffw = FeedForward(dim, dense_drop)

    def forward(
            self, 
            x: Float[torch.Tensor, "batch_size e_seq_len dim"], 
            freq: Float[torch.Tensor, "e_seq_len head_dim//2"], 
            key_padding_mask: Float[torch.Tensor, "batch_size e_seq_len"] = None
            ) -> Float[torch.Tensor, "batch_size e_seq_len dim"]:
        normed_x = self.pre_attn_norm(x)
        h = x + self.attn(normed_x, normed_x, freq, freq, key_padding_mask)
        out = h + self.ffw(self.pre_ffw_norm(h))
        return out


class DecoderBlock(torch.nn.Module):
    """
    low-level implementation of Transformer decoder block with RoPE
    """

    def __init__(self, dim: int, heads: int, attn_drop: float, dense_drop: float) -> None:
        super().__init__()
        self.pre_self_attn_norm = RMSNorm(dim)
        self.self_attn = MultiheadAttentionZero(dim, heads, attn_drop=attn_drop)
        self.pre_cross_attn_norm = RMSNorm(dim)
        self.cross_attn = MultiheadAttentionZero(dim, heads, attn_drop=attn_drop)
        self.pre_ffw_norm = RMSNorm(dim)
        self.ffw = FeedForward(dim, dense_drop)

    def forward(
        self,
        d_x: Float[torch.Tensor, "batch_size d_seq_len dim"],
        e_x: Float[torch.Tensor, "batch_size e_seq_len dim"],
        freq_d: Float[torch.Tensor, "d_seq_len head_dim//2"],
        freq_e: Float[torch.Tensor, "e_seq_len head_dim//2"],
        d_key_padding_mask: Float[torch.Tensor, "batch_size d_seq_len"],
        e_key_padding_mask: Float[torch.Tensor, "batch_size e_seq_len"],
        cache_kv: bool = False,
    ) -> torch.Tensor:
        normed_x = self.pre_self_attn_norm(d_x)
        h = d_x + self.self_attn(normed_x, normed_x, freq_d, freq_d, d_key_padding_mask, is_causal=True)
        normed_h = self.pre_cross_attn_norm(h)
        z = h + self.cross_attn(normed_h, e_x, freq_d, freq_e, e_key_padding_mask, cache_kv=cache_kv)
        out = z + self.ffw(self.pre_ffw_norm(z))
        return out
    
class T5Encoder(torch.nn.Module):
    def __init__(self, dim: int, num_layers: int, heads: int, attn_drop: float, dense_drop: float) -> None:
        super().__init__()
        self.head_dim = dim // heads
        self.layers = torch.nn.ModuleList([EncoderBlock(dim, heads, attn_drop, dense_drop) for _ in range(num_layers)])

    def forward(
            self, 
            e_x: Float[torch.Tensor, "batch_size e_seq_len dim"], 
            e_key_padding_mask: Float[torch.Tensor, "batch_size e_seq_len"]
            ) -> Float[torch.Tensor, "batch_size e_seq_len dim"]:
        """
        Args:
            e_x: (B, S, D)
            e_key_padding_mask: (B, S)
        Returns:
            x: (B, S, D)
        """
        freq = precomputed_theta_pos_frequencies(self.head_dim, e_x.shape[1]).to(e_x.device)
        for layer in self.layers:
            e_x = layer(e_x, freq, key_padding_mask=e_key_padding_mask)
        return e_x


class T5Decoder(torch.nn.Module):
    def __init__(
        self, dim: int, num_layers: int, heads: int, attn_drop: float, dense_drop: float, max_len: int = 500
    ) -> None:
        super().__init__()
        self.head_dim = dim // heads
        self.max_len = max_len
        self.layers = torch.nn.ModuleList([DecoderBlock(dim, heads, attn_drop, dense_drop) for _ in range(num_layers)])

    def forward(
        self,
        d_x: Float[torch.Tensor, "batch_size d_seq_len dim"],
        e_x: Float[torch.Tensor, "batch_size e_seq_len dim"],
        d_key_padding_mask: Float[torch.Tensor, "batch_size d_seq_len"],
        e_key_padding_mask: Float[torch.Tensor, "batch_size e_seq_len"],
        cache_kv: bool = False,
        decoding_pos: int = None,
    ) -> Float[torch.Tensor, "batch_size d_seq_len dim"]:
        """
        Args:
            d_x: (B, T, D)
            e_x: (B, S, D)
            d_key_padding_mask: (B, T)
            e_key_padding_mask: (B, S)
            cache_kv: bool
        Returns:
            x: (B, T, D)
        """
        freq = precomputed_theta_pos_frequencies(self.head_dim, self.max_len).to(d_x.device)
        if decoding_pos is not None:
            freq_d = freq[:decoding_pos]
        else:
            freq_d = freq[: d_x.shape[1]]
        freq_e = freq[: e_x.shape[1]]
        for layer in self.layers:
            d_x = layer(d_x, e_x, freq_d, freq_e, d_key_padding_mask, e_key_padding_mask, cache_kv=cache_kv)
        return d_x

class T5_3DTransformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_atoms: int,
        hidden_dim: int,
        num_heads: int = 8,
        n_layers: int = 8,
        attn_drop: float = 0.0,
        dense_drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.seq_embedder = torch.nn.Embedding(vocab_size, hidden_dim)
        self.coords_embedder = torch.nn.Linear(num_atoms * 3, hidden_dim, bias=False)
        self.encoder = T5Encoder(
            dim=hidden_dim,
            num_layers=n_layers,
            heads=num_heads,
            attn_drop=attn_drop,
            dense_drop=dense_drop,
        )
        self.decoder = T5Decoder(
            dim=hidden_dim,
            num_layers=n_layers // 2,
            heads=num_heads,
            attn_drop=attn_drop,
            dense_drop=dense_drop,
        )
        self.seq_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, vocab_size, bias=False),
        )
        self.coords_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_atoms * 3, bias=False),
        )

    def forward(self, input_data: InFrameInput) -> torch.Tensor:
        encoder_seq = input_data.EncoderSequence  # (B*4, S)
        encoder_coords = input_data.EncoderSequence # (B*4, S, N*3)
        encoder_mask = input_data.Encodermask  # (B*4, S)
        decoder_seq = input_data.DecoderSequence
        decoder_coords = input_data.Decodercoords
        decoder_mask = input_data.Decodermask

        e_x = self.seq_embedder(encoder_seq) + self.coords_embedder(encoder_coords)
        d_x = self.seq_embedder(decoder_seq) + self.coords_embedder(decoder_coords)

        e_x = self.encoder(e_x, encoder_mask.repeat_interleave(4, dim=0))
        d_x = self.decoder(d_x, e_x, decoder_mask, encoder_mask)

        logits = self.seq_predictor(d_x)
        coords_in_frame = self.coords_predictor(d_x)
        return InFrameOutput(logits, coords_in_frame)

    