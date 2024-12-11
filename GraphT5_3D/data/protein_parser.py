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
class Protein:
    name: str
    sequence: str
    backbone_N: Float[torch.Tensor, "num_res 3"]
    backbone_CA: Float[torch.Tensor, "num_res 3"]
    backbone_C: Float[torch.Tensor, "num_res 3"]

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
        seq = ""
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
            seq += res.name
        backbone_N = backbone_N[keep]
        backbone_CA = backbone_CA[keep]
        backbone_C = backbone_C[keep]
        assert len(seq) == len(backbone_CA)
        full_seq.append(seq)
        N_coords.append(backbone_N)
        CA_coords.append(backbone_CA)
        C_coords.append(backbone_C)
    
    return Protein(
        name=pdb_file.pdb_path.stem,
        sequence="".join(full_seq),
        backbone_N=torch.cat(N_coords),
        backbone_CA=torch.cat(CA_coords),
        backbone_C=torch.cat(C_coords)
    )

        
        


    
