import logging
import torch
from typing import List
from GraphT5_3D.data.constants import VOCAB
from GraphT5_3D.data.protein import Protein
from GraphT5_3D.nn.model import TransformerInput


class ProteinTokenizer:
    def __init__(self, num_extra_tokens: int = 2):
        if num_extra_tokens < 2:
            logging.warning("num_extra_tokens should be at least 2")
        self.num_extra_tokens = max(2, num_extra_tokens)
        self.vocab = (
            ["<cls>", "<pad>", "<unk>"]
            + VOCAB
            + [f"<extra_id_{i}>" for i in range(self.num_extra_tokens)]
        )
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self.pad = self.token_to_id["<pad>"]
        self.cls = self.token_to_id["<cls>"]
        self.unk = self.token_to_id["<unk>"]

    def encode(
        self, proteins: List[Protein], left_padding: bool = False, max_len: int = 1024
    ) -> TransformerInput:
        length = min(max((len(protein) for protein in proteins)), max_len - 1)
        encoded_sequence = torch.full(
            (len(proteins), length + 1), self.token_to_id["<pad>"], dtype=torch.int64
        )
        encoded_padding_mask = torch.zeros(len(proteins), length + 1)
        encoded_coords = torch.zeros(len(proteins), length + 1, 3, 3)
        encoded_common_token_mask = torch.zeros(len(proteins), length + 1)
        for i, protein in enumerate(proteins):
            if len(protein) > length:
                protein = protein.cutoff(length)
            sequence = protein.sequence
            backbones = protein.backbones
            common_token_mask = protein.common_res_mask
            if left_padding:
                offset = length - len(sequence)
            else:
                offset = 0
            encoded_sequence[i, offset] = self.token_to_id["<cls>"]
            encoded_coords[i, offset] = torch.zeros(3, 3)
            for j, res in enumerate(sequence):
                encoded_sequence[i, offset + 1 + j] = self.token_to_id.get(
                    res, self.token_to_id["<unk>"]
                )
            encoded_padding_mask[i, offset : offset + 1 + len(sequence)] = 1
            encoded_coords[i, offset + 1 : offset + 1 + len(backbones)] = backbones
            encoded_common_token_mask[
                i, offset + 1 : offset + 1 + len(common_token_mask)
            ] = common_token_mask

        return TransformerInput(
            sequence=encoded_sequence,
            coords=encoded_coords,
            padding_mask=encoded_padding_mask,
            common_token_mask=encoded_common_token_mask,
        )
