import pytest
from pathlib import Path
import torch
from GraphT5_3D.data.dataset import ProteinDataset
from GraphT5_3D.data.protein import Protein

FIXTURE_DIR = Path(__file__).resolve().parent.parent.resolve() / "fixtures"


@pytest.fixture
def mock_protein():
    return Protein(
        name="test",
        sequence=["A", "D", "C", "E", "F"],
        backbones=torch.randn(5, 3, 3),
        common_res_mask=torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]),
    )


@pytest.fixture
def mock_dataset():
    return ProteinDataset(
        config={
            "dispatcher": {"name": "from_dir", "from_dir": FIXTURE_DIR},
            "tokenizer": {
                "num_extra_tokens": 3,
            },
            "max_len": 20,
            "max_span_length": 5,
        }
    )


def test_dataset_generate_mask(mock_dataset):
    mask_regions = mock_dataset.generate_mask(20)
    assert len(mask_regions) <= 2
    assert len(mask_regions) >= 1
    assert all([len(region) == 2 for region in mask_regions])
    assert all([region[0] + region[1] <= 20 for region in mask_regions])
    assert all([region[1] <= 5 for region in mask_regions])
    for i, region in enumerate(mask_regions[:-1]):
        assert region[0] + region[1] <= mask_regions[i + 1][0]


def test_dataset_apply_mask(mock_protein, mock_dataset):
    mask_regions = [[2, 2]]
    encoder, decoder = mock_dataset.apply_mask(mock_protein, mask_regions)
    assert encoder.sequence == ["A", "D", "<extra_id_0>", "F"]
    assert decoder.sequence == ["<extra_id_0>", "C", "E", "<extra_id_1>"]
    assert torch.allclose(encoder.backbones[2], torch.zeros(3, 3))
    assert torch.allclose(decoder.backbones[0], torch.zeros(3, 3))
    assert torch.allclose(decoder.backbones[-1], torch.zeros(3, 3))
    assert torch.allclose(encoder.common_res_mask, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(decoder.common_res_mask, torch.tensor([0.0, 1.0, 0.0, 0.0]))
