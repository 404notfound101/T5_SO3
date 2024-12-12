from __future__ import annotations
import gemmi
import torch
from dataclasses import dataclass
from jaxtyping import Float
from pathlib import Path
import gzip
from typing import List

@dataclass
class ProteinFile:
    pdb_path: str
    chains: List[str]

@dataclass
class Residue:
    name: str
    N: Float[torch.Tensor, "3"]
    CA: Float[torch.Tensor, "3"]
    C: Float[torch.Tensor, "3"]
    
    def __repr__(self):
        return f"{self.name} - N: {self.N}, CA: {self.CA}, C: {self.C}"

@dataclass
class Protein:
    name: str
    sequence: List[str]
    backbones: Float[torch.Tensor, "num_res 3 3"]
    common_res_mask: Float[torch.Tensor, "num_res"]

    def __post_init__(self):
        assert len(self.sequence) == len(self.backbones), "Length mismatch"
    
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, key: int | slice) -> Residue | Protein:
        if isinstance(key, slice):
            return Protein(
                name=self.name,
                sequence=self.sequence[key],
                backbones=self.backbones[key],
                common_res_mask=self.common_res_mask[key]
            )
        return Residue(
            name=self.sequence[key],
            N=self.backbones[key][0],
            CA=self.backbones[key][1],
            C=self.backbones[key][2]
        )
    
    def cutoff(self, length: int) -> Protein:
        if len(self) <= length:
            return self
        return Protein(
            name=self.name,
            sequence=self.sequence[:length],
            backbones=self.backbones[:length],
            common_res_mask=self.common_res_mask[:length]
        )
    
    def __repr__(self):
        return f"{self.name} - {''.join(self.sequence)}"

def parse_pdb(pdb_file: ProteinFile) -> Protein:
    file_path = Path(pdb_file.pdb_path)
    if file_path.suffix == ".gz":
        pose = gemmi.read_pdb_string(gzip.open(pdb_file, mode="rt").read())
    elif file_path.suffix == ".pdb":
        pose = gemmi.read_structure(pdb_file)
    elif file_path.suffix == ".cif":
        pose = gemmi.read_structure(pdb_file)
    else:
        raise ValueError("Unsupported file format")
    
    model = pose[0]
    N_coords = []
    CA_coords = []
    C_coords = []
    full_seq = []
    for chain_id in pdb_file.chains:
        assert chain_id in model, f"Chain {chain_id} not found in the PDB file"
        chain = model[chain]
        seq = []
        backbone_N = torch.empty(len(chain), 3)
        backbone_CA = torch.empty(len(chain), 3)
        backbone_C = torch.empty(len(chain), 3)
        for i, res in enumerate(chain):
            keep = torch.ones(len(chain), dtype=torch.bool)
            if not "CA" in res:
                # skip if CA is missing
                keep[i] = False
                continue
            else:
                backbone_CA[i] = torch.tensor(res["CA"][0].pos.tolist())
            if "N" in res:
                backbone_N[i] = torch.tensor(res["N"][0].pos.tolist())
            else:
                backbone_N[i] = torch.tensor(res["CA"][0].pos.tolist())
            if "C" in res:
                backbone_C[i] = torch.tensor(res["C"][0].pos.tolist())
            else:
                backbone_C[i] = torch.tensor(res["CA"][0].pos.tolist())
            seq.append(res.name)
        backbone_N = backbone_N[keep]
        backbone_CA = backbone_CA[keep]
        backbone_C = backbone_C[keep]
        seq = gemmi.one_letter_code(seq)
        assert len(seq) == len(backbone_CA)
        full_seq += seq
        N_coords.append(backbone_N)
        CA_coords.append(backbone_CA)
        C_coords.append(backbone_C)
    
    return Protein(
        name=pdb_file.pdb_path.stem,
        sequence=full_seq,
        N=torch.cat(N_coords),
        CA=torch.cat(CA_coords),
        C=torch.cat(C_coords)
    )

        
        


    
