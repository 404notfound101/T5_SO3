from __future__ import annotations
import torch
from typing import Tuple
from jaxtyping import Float, Int64
from dataclasses import dataclass
from GraphT5_3D.nn.modules.RoPE import precomputed_theta_pos_frequencies
from GraphT5_3D.nn.modules.frame_averaging import create_frames, apply_frames, invert_frames
from GraphT5_3D.nn.modules.norm import RMSNorm
from GraphT5_3D.nn.modules.transformer import MultiheadAttentionZero, FeedForward

@dataclass
class TransformerInput:
    sequence: Int64[torch.Tensor, "batch_size seq_len"]
    coords: Float[torch.Tensor, "batch_size seq_len 3 3"]
    padding_mask: Float[torch.Tensor, "batch_size seq_len"]
    common_token_mask: Float[torch.Tensor, "batch_size seq_len"]

    def get_input_slice(self) -> TransformerInput:
        return TransformerInput(
            sequence=self.sequence[:, :-1],
            coords=self.coords[:, :-1],
            padding_mask=self.padding_mask[:, :-1],
            common_token_mask=self.common_token_mask[:, :-1]
        )
    
    def get_target_slice(self) -> TransformerInput:
        return TransformerInput(
            sequence=self.sequence[:, 1:],
            coords=self.coords[:, 1:],
            padding_mask=self.padding_mask[:, 1:],
            common_token_mask=self.common_token_mask[:, 1:]
        )

    def create_frames(self) -> Tuple[Float[torch.Tensor, "batch_size 4 3 3"], Float[torch.Tensor, "batch_size 1 3"]]:
        frames, center = create_frames(self.coords, self.common_token_mask)
        self.frames = frames
        self.center = center
        return frames, center

    def apply_frames(self, frames: Float[torch.Tensor, "batch_size 4 3 3"], center: Float[torch.Tensor, "batch_size 1 3"]) -> None:
        assert len(frames) == len(self.coords), "batch mismatch"
        framed_coords = apply_frames(self.coords, self.common_token_mask, frames, center)
        self.coords = framed_coords
        self.sequence = self.sequence.repeat_interleave(4, dim=0)
        self.padding_mask = self.padding_mask.repeat_interleave(4, dim=0)


@dataclass
class T5Input:
    encoder_input: TransformerInput
    decoder_input: TransformerInput

@dataclass
class EncoderOutput:
    encoder_embedding: Float[torch.Tensor, "fbatch_size e_seq_len dim"]
    encoder_padding_mask: Float[torch.Tensor, "batch_size e_seq_len"]
    frames: Float[torch.Tensor, "batch_size 4 3 3"]
    center: Float[torch.Tensor, "batch_size 1 3"]

@dataclass
class DecoderOutput:
    logits: Float[torch.Tensor, "fbatch_size d_seq_len vocab_size"]
    coords_in_frame: Float[torch.Tensor, "fbatch_size d_seq_len 9"]
    common_token_mask: Float[torch.Tensor, "batch_size d_seq_len"]
    frames: Float[torch.Tensor, "batch_size 4 3 3"]
    center: Float[torch.Tensor, "batch_size 1 3"]

    def invert_frames(self) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = invert_frames(
            self.coords_in_frame, 
            self.common_token_mask,
            self.frames,
            self.center).reshape(self.common_token_mask.shape + [3, 3])
        logits = self.logits.reshape([-1, 4] + self.logits.shape[1:]).mean(dim=1)
        return logits, coords


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
    def __init__(self, dim: int, num_layers: int, heads: int, attn_drop: float, dense_drop: float, max_len: int = 1000) -> None:
        super().__init__()
        self.head_dim = dim // heads
        self.layers = torch.nn.ModuleList([EncoderBlock(dim, heads, attn_drop, dense_drop) for _ in range(num_layers)])
        self.register_buffer("freq", precomputed_theta_pos_frequencies(self.head_dim, max_len))
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
        freq = self.freq[: e_x.shape[1]]
        for layer in self.layers:
            e_x = layer(e_x, freq, key_padding_mask=e_key_padding_mask)
        return e_x


class T5Decoder(torch.nn.Module):
    def __init__(
        self, dim: int, num_layers: int, heads: int, attn_drop: float, dense_drop: float, max_len: int = 1000
    ) -> None:
        super().__init__()
        self.head_dim = dim // heads
        self.max_len = max_len
        self.layers = torch.nn.ModuleList([DecoderBlock(dim, heads, attn_drop, dense_drop) for _ in range(num_layers)])
        self.register_buffer("freq", precomputed_theta_pos_frequencies(self.head_dim, self.max_len))

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
        if decoding_pos is not None:
            freq_d = self.freq[:decoding_pos]
        else:
            freq_d = self.freq[: d_x.shape[1]]
        freq_e = self.freq[: e_x.shape[1]]
        for layer in self.layers:
            d_x = layer(d_x, e_x, freq_d, freq_e, d_key_padding_mask, e_key_padding_mask, cache_kv=cache_kv)
        return d_x

class T5_3DTransformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_heads: int = 8,
        n_layers: int = 8,
        attn_drop: float = 0.0,
        dense_drop: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.seq_embedder = torch.nn.Embedding(vocab_size, hidden_dim)
        self.coords_embedder = torch.nn.Linear(9, hidden_dim, bias=False)
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
            torch.nn.Linear(hidden_dim, 9, bias=False),
        )

    def forward(self, input_data: T5Input) -> DecoderOutput:
        encoder_output = self.encode(input_data.encoder_input)
        output = self.decode(
            input_data.decoder_input, encoder_output.encoder_embedding, input_data.encoder_input.padding_mask
        )
        return output
    
    def encode(self, encoder_input: TransformerInput) -> EncoderOutput:
        frames, center = encoder_input.create_frames()
        encoder_input.apply_frames(frames, center)
        encoder_seq = encoder_input.sequence
        encoder_coords = encoder_input.coords
        encoder_mask = encoder_input.padding_mask

        e_x = self.seq_embedder(encoder_seq) + self.coords_embedder(encoder_coords)
        e_x = self.encoder(e_x, encoder_mask)
        return EncoderOutput(
            encoder_embedding=e_x,
            encoder_padding_mask=encoder_mask,
            frames=frames,
            center=center,
        )
    
    def decode(
            self, 
            decoder_input: TransformerInput, 
            encoder_output: EncoderOutput,
            cache_kv: bool = False,
            decoding_pos: int = None,
            ) -> Float[torch.Tensor, "batch_size d_seq_len vocab_size"]:
        frames = encoder_output.frames
        center = encoder_output.center
        encoder_embedding = encoder_output.encoder_embedding
        encoder_padding_mask = encoder_output.encoder_padding_mask
        decoder_input.apply_frames(frames, center)

        decoder_seq = decoder_input.sequence
        decoder_coords = decoder_input.coords
        decoder_mask = decoder_input.padding_mask

        d_x = self.seq_embedder(decoder_seq) + self.coords_embedder(decoder_coords)
        d_x = self.decoder(d_x, encoder_embedding, decoder_mask, encoder_padding_mask, cache_kv=cache_kv, decoding_pos=decoding_pos)
        logits = self.seq_predictor(d_x)
        coords_in_frame = self.coords_predictor(d_x)
        return DecoderOutput(
            logits=logits,
            coords_in_frame=coords_in_frame,
            common_token_mask=decoder_input.common_token_mask,
            frames=frames,
            center=center,
        )

    