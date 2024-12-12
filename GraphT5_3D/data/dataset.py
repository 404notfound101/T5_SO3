import torch
import random
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from GraphT5_3D.data.dispatcher import DispatcherFactory
from GraphT5_3D.data.tokenizer import ProteinTokenizer
from GraphT5_3D.data.protein import Protein, parse_pdb
from GraphT5_3D.nn.model import T5Input

class ProteinDataset(Dataset):
    def __init__(
            self,
            config: Dict,
    ) -> None:
        self.dispatcher = DispatcherFactory.get_dispatcher(config["dispatcher"]["name"])(**config["dispatcher"])
        self.tokenizer = ProteinTokenizer(**config["tokenizer"])
        self.max_len = config.get("max_len", 1024)
        self.min_span_length = config.get("min_span_length", 1)
        self.max_span_length = config.get("max_span_length", 10)
    
    def __len__(self) -> int:
        return len(self.dispatcher)
    
    def __getitem__(self, idx):
        protein = parse_pdb(self.dispatcher[idx]).cutoff(self.max_len)
        mask_regions = self.random_mask(protein)
        encoder_protein, decoder_protein = self.apply_mask(protein, mask_regions)
        return [encoder_protein, decoder_protein]


    def random_mask(self, seq_len: int) -> List[List[int]]:
        """
        Randomly generate mask regions
        Args:
            seq_len(int): length of the sequence
            min_span_length(int): minimum length of the mask spans
            max_span_length(int): maximum length of the mask spans
        Returns:
            mask_regions(List[List[int]]): ordered list of mask regions, each region is a list of two integers [start, length]
        """
        num_spans = random.randint(1, self.tokenizer.num_extra_tokens - 1)
        region_len = seq_len // num_spans
        mask_regions = []
        for i in range(num_spans):
            region_start = region_len * i
            region_end = region_len * (i + 1)
            span_length = random.randint(self.min_span_length, min(self.max_span_length, region_len))
            start = random.randint(region_start, region_end - span_length)
            mask_regions.append([start, span_length])
        return mask_regions

    def apply_mask(self, protein: Protein, mask_regions: List[List[int]]) -> Tuple[Protein]:
        """
        T5-style masking of the input sequence.
        E.g. Thank you for inviting me to your party last week.
        -> Thank you <x> me to your party <y> week. | <x> for inviting <y> last <z>
        Args:
            protein(Protein): protein instance
            mask_regions(List[List[int]]): ordered list of mask regions, each region is a list of two integers [start, length]
        Returns:

        """
        # initialize
        generate_length = sum([region[1] for region in mask_regions])
        num_regions = len(mask_regions)
        encoder_length = len(protein) - generate_length + num_regions
        decoder_length = generate_length + num_regions + 1
        encoder_coords = torch.zeros(encoder_length, 3, 3)
        decoder_coords = torch.zeros(decoder_length, 3, 3)
        encoder_common_token_mask = torch.zeros(encoder_length)
        decoder_common_token_mask = torch.zeros(decoder_length)
        encoder_seq = []
        decoder_seq = []

        # assign regions
        input_pointer = 0
        encoder_pointer = 0
        decoder_pointer = 0
        for i, region in enumerate(mask_regions):
            start, length = region
            # encoder
            encoder_part_length = start - input_pointer
            encoder_seq += protein.sequence[input_pointer: start]
            encoder_coords[encoder_pointer: encoder_pointer + encoder_part_length] = protein.backbones[input_pointer: start]
            encoder_pointer += encoder_part_length
            encoder_seq += [f"<extra_id_{i}>"]
            encoder_coords[encoder_pointer] = torch.zeros(3, 3)
            encoder_common_token_mask[encoder_pointer] = 1
            encoder_pointer += 1
            # decoder
            decoder_seq += [f"<extra_id_{i}>"]
            decoder_coords[decoder_pointer] = torch.zeros(3, 3)
            decoder_common_token_mask[decoder_pointer] = 1
            decoder_pointer += 1
            decoder_seq += protein.sequence[start: start + length]
            decoder_coords[decoder_pointer: decoder_pointer + length] = protein.backbones[start: start + length]
            decoder_pointer += length
            input_pointer = start + length
        decoder_seq += [f"<extra_id_{i+1}>"]
        decoder_coords[decoder_pointer] = torch.zeros(3, 3)
        decoder_common_token_mask[decoder_pointer] = 1

        encoder_protein = Protein(
            name=f"{protein.name}_encoder",
            sequence=encoder_seq,
            backbones=encoder_coords,
            common_res_mask=encoder_common_token_mask,
        )
        decoder_protein = Protein(
            name=f"{protein.name}_decoder",
            sequence=decoder_seq,
            backbones=decoder_coords,
            common_res_mask=decoder_common_token_mask,
        )
        return encoder_protein, decoder_protein

    def collate_fn(self, batch: List[List[Protein]]) -> T5Input:
        encoder_inputs = self.tokenizer.encode([protein[0] for protein in batch], left_padding=False, max_len=self.max_len)
        decoder_inputs = self.tokenizer.encode([protein[1] for protein in batch], left_padding=True, max_len=self.max_len)
        return T5Input(encoder_input=encoder_inputs, decoder_input=decoder_inputs)
