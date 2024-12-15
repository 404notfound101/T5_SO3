import pytest
import torch
from GraphT5_3D.data.tokenizer import ProteinTokenizer
from GraphT5_3D.data.protein import Protein
from typing import List


@pytest.fixture
def mock_proteins_encoder() -> List[Protein]:
    return [
        Protein(
            name="test",
            sequence=["A", "<extra_id_0>", "C", "<extra_id_1>"],
            backbones=torch.randn(4, 3, 3),
            common_res_mask=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        ),
        Protein(
            name="test2",
            sequence=["D", "E", "F"],
            backbones=torch.randn(3, 3, 3),
            common_res_mask=torch.ones(3),
        ),
    ]


@pytest.fixture
def mock_proteins_decoder() -> List[Protein]:
    return [
        Protein(
            name="test",
            sequence=["<extra_id_0>", "C", "D", "<extra_id_1>"],
            backbones=torch.randn(4, 3, 3),
            common_res_mask=torch.tensor([0, 1, 1, 0]),
        ),
        Protein(
            name="test2",
            sequence=["<extra_id_0>", "C", "<extra_id_1>"],
            backbones=torch.randn(3, 3, 3),
            common_res_mask=torch.tensor([0, 1, 0]),
        ),
    ]


def test_protein_encoder(mock_proteins_encoder):
    tokenizer = ProteinTokenizer()
    encoded = tokenizer.encode(mock_proteins_encoder, left_padding=False, max_len=10)
    assert encoded.sequence.shape == torch.Size([2, 5])
    assert encoded.coords.shape == torch.Size([2, 5, 3, 3])
    assert torch.allclose(
        encoded.sequence, torch.tensor([[0, 4, 28, 22, 29], [0, 12, 8, 17, 1]])
    )
    assert torch.allclose(encoded.coords[:, 0], torch.zeros(2, 3, 3))  # CLS token
    assert torch.allclose(encoded.coords[1, -1], torch.zeros(3, 3))  # PAD token
    assert torch.allclose(
        encoded.common_token_mask,
        torch.tensor([[0, 1, 0, 1, 0], [0, 1, 1, 1, 0]], dtype=torch.float32),
    )


def test_protein_decoder(mock_proteins_decoder):
    tokenizer = ProteinTokenizer()
    encoded = tokenizer.encode(mock_proteins_decoder, left_padding=True, max_len=10)
    print(encoded.sequence)
    assert encoded.sequence.shape == torch.Size([2, 5])
    assert encoded.coords.shape == torch.Size([2, 5, 3, 3])
    assert torch.allclose(
        encoded.sequence, torch.tensor([[0, 28, 22, 12, 29], [1, 0, 28, 22, 29]])
    )
    assert torch.allclose(encoded.coords[0, 0], torch.zeros(3, 3))  # CLS token
    assert torch.allclose(encoded.coords[1, 1], torch.zeros(3, 3))  # CLS token
    assert torch.allclose(encoded.coords[1, 0], torch.zeros(3, 3))  # PAD token
    assert torch.allclose(
        encoded.common_token_mask,
        torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 1, 0]], dtype=torch.float32),
    )
