import pytest
import os
import torch
from GraphT5_3D.data.protein import Protein, ProteinFile, parse_pdb
from pathlib import Path
from typing import List
from glob import glob

FIXTURE_DIR = Path(__file__).resolve().parent.parent.resolve() / "fixtures"


@pytest.fixture
def mock_protein() -> Protein:
    return Protein(
        name="test",
        sequence=["A", "D", "C", "E", "F"],
        backbones=torch.randn(5, 3, 3),
        common_res_mask=torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]),
    )


@pytest.fixture
def protein_files() -> List[ProteinFile]:
    files = glob(os.path.join(FIXTURE_DIR, "*"))
    return [ProteinFile(pdb_path=f, chains=["B"]) for f in files]


def test_protein(mock_protein):
    assert len(mock_protein) == 5
    assert mock_protein[0].name == "A"
    assert mock_protein[1:3].sequence == ["D", "C"]
    assert mock_protein[1:3].backbones.shape == torch.Size([2, 3, 3])
    assert torch.allclose(mock_protein[1:3].common_res_mask, torch.tensor([0.0, 1.0]))
    assert mock_protein.cutoff(2).sequence == ["A", "D"]


def test_parse_pdb(protein_files):
    for file in protein_files:
        protein = parse_pdb(file)
        assert protein.sequence[:3] == ["V", "Q", "L"]
        assert protein.backbones.shape == (120, 3, 3)
        assert len(protein) == 120
