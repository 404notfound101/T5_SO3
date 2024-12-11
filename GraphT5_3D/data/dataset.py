import torch

class FrameworkEncoderDataset(RediffDatasetBase):
    """
    Dataset class for pre-training the framework encoder in a T5 encoder-decoder workflow
    """

    def __init__(
        self,
        data_dir: str,
        metafile: str = None,
        num_sentinel_tokens: int = 2,
        pad_token_id: int = 20,
        eos_token_id: int = 21,
        ignore_idx: int = -100,
        min_span_length: int = 5,
        max_span_length: int = 25,
        H3_learning_weights: float = 0.0,
        get_path: Callable = None,
        eval=False,
        **kwargs,
    ) -> None:
        super().__init__(data_pipeline=FrameworkEncoderDataProcessing.from_path(data_dir, **kwargs))
        self.meta = pd.read_csv(metafile) if metafile else None
        self.data_dir = data_dir
        self.sentinel_tokens = torch.arange(num_sentinel_tokens) + 22
        self.padding_token = pad_token_id
        self.eos_token = eos_token_id
        self.ignore_idx = ignore_idx
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.H3_learning_weights = H3_learning_weights
        self.eval = eval
        if get_path is not None:
            self._get_path = get_path.__get__(self, FrameworkEncoderDataset)

    def __len__(self) -> int:
        if self.meta is not None:
            return len(self.meta)
        return len(self.data_pipeline.src)

    def _get_path(self, idx: int) -> str:
        if self.meta is not None:
            return os.path.join(self.data_dir, self.meta["prefix"][idx] + ".pdb")
        return os.path.join(self.data_dir, self.data_pipeline.src[idx])

    def H3_mask(self, cdr_flags: torch.Tensor) -> List[List[int]]:
        """
        Generate mask for H3 region
        Args:
            cdr_flags(torch.Tensor): tensor indicating the CDR regions
        Returns:
            output(torch.Tensor): mask tensor for H3 region
        """
        start = torch.argmax(cdr_flags).int().item()
        mask_regions = [[start, cdr_flags.sum().int().item()]]
        return mask_regions

    def random_mask(self, seq_len: int) -> Tuple[List[int], torch.Tensor]:
        """
        Randomly generate mask regions
        Args:
            seq_len(int): length of the sequence
            min_span_length(int): minimum length of the mask spans
            max_span_length(int): maximum length of the mask spans
        Returns:
            mask_regions(List[List[int]]): list of mask regions, each region is a list of two integers [start, length]
        """
        num_spans = random.randint(1, len(self.sentinel_tokens) - 1)
        region_len = seq_len // num_spans
        mask = torch.zeros(seq_len)
        mask_regions = []
        for i in range(num_spans):
            region_start = region_len * i
            region_end = region_len * (i + 1)
            span_length = random.randint(self.min_span_length, min(self.max_span_length, region_len))
            start = random.randint(region_start, region_end - span_length)
            mask[start : start + span_length] = 1
            mask_regions.append([start, span_length])
        return mask_regions, mask

    def apply_mask(self, mask_regions, source_seq: torch.Tensor, source_pos: torch.Tensor):
        """
        T5-style masking of the input sequence.
        E.g. Thank you for inviting me to your party last week.
        -> Thank you <x> me to your party <y> week. | <x> for inviting <y> last <z>
        Args:
            mask_regions(List[List[int]]): list of mask regions, each region is a list of two integers [start, length]
            source_seq(torch.Tensor): source sequence tensor
            source_pos(torch.Tensor): source position tensor
        Returns:
            output(Dict[str, torch.Tensor]): dictionary containing encoder and decoder input tensors
            encoder_seq_input(torch.Tensor): encoder input sequence tensor
            encoder_pos_input(torch.Tensor): encoder input position tensor, zeros for masked regions
            decoder_seq_input(torch.Tensor): decoder input sequence tensor
            decoder_target(torch.Tensor): decoder target sequence tensor, shifted by one from decoder_seq_input, last token will be ignored
            encoder_source_mask(torch.Tensor): mask tensor to slice the non-corrupted regions from the source sequence
            encoder_target_mask(torch.Tensor): mask tensor to slice the non-corrupted regions from the encoder input
        """
        seq_len = source_seq.shape[0]
        num_mask_spans = len(mask_regions)
        if num_mask_spans >= len(self.sentinel_tokens):
            raise ValueError("Number of mask spans must be smaller than the number of sentinel tokens")
        total_mask_len = sum([pos[1] for pos in mask_regions])
        encoder_input_length = seq_len - total_mask_len + num_mask_spans + 1  # for eos token
        decoder_input_length = total_mask_len + num_mask_spans + 2  # for last sentinel token and start padding token

        encoder_seq_input = torch.zeros(encoder_input_length, dtype=source_seq.dtype)
        encoder_pos_input = torch.zeros(encoder_input_length, *source_pos.shape[1:], dtype=source_pos.dtype)
        decoder_seq_input = torch.zeros(decoder_input_length, dtype=source_seq.dtype)
        decoder_target = torch.zeros(decoder_input_length, dtype=torch.int64)
        source_mask = torch.ones(seq_len, dtype=torch.bool)
        target_mask = torch.ones(encoder_input_length, dtype=torch.bool)
        target_mask[-1] = False
        shorten = 0
        decoder_start = 1
        decoder_seq_input[0] = self.padding_token
        for i, pos in enumerate(mask_regions):
            start = pos[0]
            length = pos[1]
            source_mask[start : start + length] = False
            target_mask[start - shorten] = False
            encoder_seq_input[start - shorten] = self.sentinel_tokens[i]
            decoder_seq_input[decoder_start] = self.sentinel_tokens[i]
            decoder_seq_input[decoder_start + 1 : decoder_start + 1 + length] = source_seq[start : start + length]
            decoder_start += length + 1
            shorten += length - 1
        encoder_seq_input[target_mask] = source_seq[source_mask]
        encoder_seq_input[-1] = self.eos_token
        encoder_pos_input[target_mask] = source_pos[source_mask]
        decoder_seq_input[-1] = self.sentinel_tokens[i + 1]
        decoder_target[:-1] = decoder_seq_input[1:]
        decoder_target[-1] = self.eos_token

        return {
            "encoder_seq_input": encoder_seq_input,
            "encoder_pos_input": encoder_pos_input,
            "decoder_seq_input": decoder_seq_input,
            "decoder_target": decoder_target,
            "encoder_source_mask": source_mask,
            "encoder_target_mask": target_mask,
        }

    def create_frame(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply frame transformation to the input coordinates to make encoder input equivariant to rotation and translation
        Args:
            coords(torch.Tensor): input coordinates tensor
            mask(torch.Tensor): mask tensor to slice the non-corrupted regions
        Returns:
            output(torch.Tensor): transformed coordinates tensor
        """

        x = coords[mask == 0].reshape(-1, 3)
        center = x.mean(dim=0)
        centered_x = x - center
        cov = centered_x.t() @ centered_x
        eigenvectors = torch.linalg.eigh(cov, UPLO="U")[1][:, :2]  # (3, 2)
        ops = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        frames = eigenvectors.unsqueeze(0) * ops.unsqueeze(1)  # (4, 3, 2)
        frames = torch.cat([frames, torch.linalg.cross(frames[..., 0], frames[..., 1]).unsqueeze(-1)], dim=-1)
        centered_coords = coords.reshape(-1, 3) - center
        return centered_coords.reshape(coords.shape), frames, center

    def post_process(self, parse_pose_result: ParsePoseResult) -> Dict[str, Dict[str, torch.Tensor]]:
        seq = parse_pose_result.seq
        pos = parse_pose_result.coords[:, :3]
        assert seq.shape[0] == pos.shape[0]
        if self.eval:
            mask_tensor = parse_pose_result.cdr_flags
            mask_regions = self.H3_mask(mask_tensor)
        else:
            if random.random() < self.H3_learning_weights:
                mask_tensor = parse_pose_result.cdr_flags
                mask_regions = self.H3_mask(mask_tensor)
            else:
                mask_regions, mask_tensor = self.random_mask(seq.shape[0])

        centered_coords, frames, center = self.create_frame(pos, mask_tensor)

        model_inputs = self.apply_mask(mask_regions, seq, centered_coords)
        encoder_pos_input = model_inputs["encoder_pos_input"]

        framed_x = torch.einsum("oij,sj->osi", frames.transpose(1, 2), encoder_pos_input.reshape(-1, 3))
        model_inputs["encoder_pos_input"] = framed_x.reshape(4, encoder_pos_input.shape[0], -1)
        return model_inputs

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        safe_batch = [sample for sample in batch if sample is not None]
        if len(safe_batch) == 0:
            logging.warning("Empty batch")
            return None
        batchsize = len(safe_batch)
        max_len_encoder = max([s["encoder_seq_input"].shape[0] for s in safe_batch])
        max_len_decoder = max([s["decoder_seq_input"].shape[0] for s in safe_batch])
        pos_dim = safe_batch[0]["encoder_pos_input"].shape[-1]
        encoder_seq_input = torch.full([batchsize, max_len_encoder], fill_value=self.padding_token, dtype=torch.int32)
        encoder_pos_input = torch.zeros(batchsize * 4, max_len_encoder, pos_dim)
        decoder_seq_input = torch.full([batchsize, max_len_decoder], fill_value=self.padding_token, dtype=torch.int32)
        decoder_target = torch.full([batchsize, max_len_decoder], fill_value=self.ignore_idx, dtype=torch.int64)
        encoder_mask = torch.zeros(batchsize, max_len_encoder, dtype=torch.bool)
        decoder_mask = torch.zeros(batchsize, max_len_decoder, dtype=torch.bool)
        for i, sample in enumerate(safe_batch):
            encoder_seq_input[i, : sample["encoder_seq_input"].shape[0]] = sample["encoder_seq_input"]
            encoder_pos_input[4 * i : 4 * (i + 1), : sample["encoder_pos_input"].shape[1]] = sample["encoder_pos_input"]
            decoder_seq_input[i, : sample["decoder_seq_input"].shape[0]] = sample["decoder_seq_input"]
            decoder_target[i, : sample["decoder_target"].shape[0]] = sample["decoder_target"]
            encoder_mask[i, : sample["encoder_seq_input"].shape[0]] = True
            decoder_mask[i, : sample["decoder_seq_input"].shape[0]] = True
        encoder_seq_input = encoder_seq_input.repeat_interleave(4, dim=0)

        return {
            "input": {
                "encoder_seq_input": encoder_seq_input,
                "encoder_pos_input": encoder_pos_input,
                "decoder_seq_input": decoder_seq_input,
                "encoder_mask": encoder_mask,
                "decoder_mask": decoder_mask,
            },
            "target": {
                "decoder_target": decoder_target,
            },
        }


@factory.register_dataset("EncoderDecoderDataset")
class EncoderDecoderDataset(RediffDataset):
    def post_process(self, parse_pose_result: ParsePoseResult) -> Dict[str, Dict[str, torch.Tensor]]:
        # forward noising
        t = random.randint(1, self.num_steps)
        t_embed = t * parse_pose_result.cdr_flags
        _, noised_seq = self.categorical_noiser.add_noise(
            x_0=parse_pose_result.seq,
            mask_generate=parse_pose_result.cdr_flags,
            t=t,
        )
        posterior = self.categorical_noiser.posterior(noised_seq, parse_pose_result.seq, t=t_embed)
        noised_coords, eps = self.gaussian_noiser.add_noise(
            p_0=parse_pose_result.coords,
            mask_generate=parse_pose_result.cdr_flags,
            t=t,
        )
        noised_pos, noised_residue_features = seperate_coord(noised_coords)

        return {
            "input": {
                "encoder_seq": noised_seq[parse_pose_result.cdr_flags == 0],
                "encoder_pos": noised_pos[parse_pose_result.cdr_flags == 0],
                "encoder_residue_features": noised_residue_features[parse_pose_result.cdr_flags == 0],
                "encoder_entity_tags": parse_pose_result.entity_tags[parse_pose_result.cdr_flags == 0],
                "encoder_t_embed": t_embed[parse_pose_result.cdr_flags == 0],
                "decoder_seq": noised_seq[parse_pose_result.cdr_flags == 1],
                "decoder_pos": noised_pos[parse_pose_result.cdr_flags == 1],
                "decoder_residue_features": noised_residue_features[parse_pose_result.cdr_flags == 1],
                "decoder_entity_tags": parse_pose_result.entity_tags[parse_pose_result.cdr_flags == 1],
                "decoder_t_embed": t_embed[parse_pose_result.cdr_flags == 1],
            },
            "target": {
                "seq_posterior": posterior[parse_pose_result.cdr_flags == 1],
                "eps": eps[parse_pose_result.cdr_flags == 1],
                "generate_mask": parse_pose_result.cdr_flags,
            },
        }

    def collate_fn(self, batch: List[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
        safe_batch = [sample for sample in batch if sample is not None]
        if len(safe_batch) == 0:
            logging.warning("Empty batch")
            return None

        max_len_decoder = max([s["input"]["decoder_seq"].shape[0] for s in safe_batch])
        max_len_encoder = max([s["input"]["encoder_seq"].shape[0] for s in safe_batch])
        num_atoms_residue_features = safe_batch[0]["input"]["decoder_residue_features"].shape[1]
        num_classes = safe_batch[0]["target"]["seq_posterior"].shape[-1]

        minibatch = []
        for i, s in enumerate(safe_batch):
            minibatch.append(torch.ones(max_len_decoder, dtype=torch.int64) * i)
        batch = torch.cat(minibatch, dim=0)

        batch_seq_encoder = torch.zeros(len(safe_batch), max_len_encoder, dtype=torch.int64)
        batch_pos_encoder = torch.zeros(len(safe_batch), max_len_encoder, 3)
        batch_residue_features_encoder = torch.zeros(len(safe_batch), max_len_encoder, num_atoms_residue_features, 3)
        batch_entity_tags_encoder = torch.zeros(len(safe_batch), max_len_encoder, dtype=torch.int64)
        batch_t_embed_encoder = torch.zeros(len(safe_batch), max_len_encoder, dtype=torch.int64)
        batch_seq_decoder = torch.zeros(len(safe_batch), max_len_decoder, dtype=torch.int64)
        batch_pos_decoder = torch.zeros(len(safe_batch), max_len_decoder, 3)
        batch_residue_features_decoder = torch.zeros(len(safe_batch), max_len_decoder, num_atoms_residue_features, 3)
        batch_entity_tags_decoder = torch.zeros(len(safe_batch), max_len_decoder, dtype=torch.int64)
        batch_t_embed_decoder = torch.zeros(len(safe_batch), max_len_decoder, dtype=torch.int64)
        batch_target_seq = torch.zeros(len(safe_batch), max_len_decoder, dtype=torch.int64)
        batch_target_seq = torch.nn.functional.one_hot(batch_target_seq, num_classes).float()
        batch_target_eps = torch.zeros(len(safe_batch), max_len_decoder, num_atoms_residue_features + 1, 3)
        batch_generate_mask = torch.zeros(len(safe_batch), max_len_decoder)
        encoder_mask = torch.zeros(len(safe_batch), max_len_encoder)
        decoder_mask = torch.zeros(len(safe_batch), max_len_decoder)

        for i, s in enumerate(safe_batch):
            encoder_seq_len = s["input"]["encoder_seq"].shape[0]
            decoder_seq_len = s["input"]["decoder_seq"].shape[0]
            batch_seq_encoder[i, :encoder_seq_len] = s["input"]["encoder_seq"]
            batch_pos_encoder[i, :encoder_seq_len] = s["input"]["encoder_pos"]
            batch_residue_features_encoder[i, :encoder_seq_len] = s["input"]["encoder_residue_features"]
            batch_entity_tags_encoder[i, :encoder_seq_len] = s["input"]["encoder_entity_tags"]
            batch_t_embed_encoder[i, :encoder_seq_len] = s["input"]["encoder_t_embed"]
            batch_seq_decoder[i, :decoder_seq_len] = s["input"]["decoder_seq"]
            batch_pos_decoder[i, :decoder_seq_len] = s["input"]["decoder_pos"]
            batch_residue_features_decoder[i, :decoder_seq_len] = s["input"]["decoder_residue_features"]
            batch_entity_tags_decoder[i, :decoder_seq_len] = s["input"]["decoder_entity_tags"]
            batch_t_embed_decoder[i, :decoder_seq_len] = s["input"]["decoder_t_embed"]
            batch_target_seq[i, :decoder_seq_len] = s["target"]["seq_posterior"]
            batch_target_eps[i, :decoder_seq_len] = s["target"]["eps"]
            encoder_mask[i, :encoder_seq_len] = 1
            decoder_mask[i, :decoder_seq_len] = 1

        batch_seq_decoder = batch_seq_decoder.reshape(-1)
        batch_pos_decoder = batch_pos_decoder.reshape((-1,) + batch_pos_decoder.shape[2:])
        batch_residue_features_decoder = batch_residue_features_decoder.reshape(
            (-1,) + batch_residue_features_decoder.shape[2:]
        )
        batch_t_embed_decoder = batch_t_embed_decoder.reshape(-1)
        batch_target_seq = batch_target_seq.reshape((-1,) + batch_target_seq.shape[2:])
        batch_target_eps = batch_target_eps.reshape((-1,) + batch_target_eps.shape[2:])
        batch_generate_mask = decoder_mask.reshape(-1)
        encoder_pos_input = torch.cat([batch_pos_encoder.unsqueeze(2), batch_residue_features_encoder], dim=2)

        return {
            "input": {
                "encoder_seq_input": batch_seq_encoder,  # (B, S)
                "encoder_pos_input": encoder_pos_input,  # (B, S, N + 1, 3)
                "encoder_entity_tags": batch_entity_tags_encoder,  # (B, S)
                "encoder_t_embed": batch_t_embed_encoder,  # (B, S)
                "seq": batch_seq_decoder,  # (B * T)
                "pos": batch_pos_decoder,  # (B * T, 3)
                "residue_features": batch_residue_features_decoder,  # (B * T, N, 3)
                "entity_tags": batch_entity_tags_decoder,  # (B, T)
                "t_embed": batch_t_embed_decoder,  # (B * T)
                "batch": batch,  # (B * T)
                "encoder_mask": encoder_mask,  # (B, S)
                "decoder_mask": decoder_mask,  # (B, T)
            },
            "target": {
                "seq_xt": batch_seq_decoder,  # (B * T)
                "seq_posterior": batch_target_seq,  # (B * T, C)
                "eps": batch_target_eps,  # (B * T, N + 1, 3)
                "t_embed": batch_t_embed_decoder,  # (B * T)
                "generate_mask": batch_generate_mask,  # (B * T)
            },
        }


class EncoderDecoderDatasetSampling(RediffDatasetSampling):
    def post_process(self, parse_pose_result: ParsePoseResult) -> Dict[str, Dict[str, torch.Tensor]]:
        # replace the design region to prior
        input_data = self.data_pipeline.batch(parse_pose_result, self.samples_in_batch)
        generate_mask = input_data["generate_mask"]
        seq_encoder = input_data["seq"][generate_mask == 0].reshape(self.samples_in_batch, -1)
        pos_encoder = input_data["pos"][generate_mask == 0].reshape(
            self.samples_in_batch, -1, *input_data["pos"].shape[1:]
        )
        residue_features_encoder = input_data["residue_features"][generate_mask == 0].reshape(
            self.samples_in_batch, -1, *input_data["residue_features"].shape[1:]
        )
        entity_tags_encoder = input_data["entity_tags"][generate_mask == 0].reshape(self.samples_in_batch, -1)
        seq_docoder = input_data["seq"][generate_mask == 1].reshape(self.samples_in_batch, -1)
        pos_decoder = input_data["pos"][generate_mask == 1].reshape(
            self.samples_in_batch, -1, *input_data["pos"].shape[1:]
        )
        residue_features_decoder = input_data["residue_features"][generate_mask == 1].reshape(
            self.samples_in_batch, -1, *input_data["residue_features"].shape[1:]
        )
        entity_tags_decoder = input_data["entity_tags"][generate_mask == 1].reshape(self.samples_in_batch, -1)
        t_embed_encoder = torch.zeros_like(seq_encoder)

        return {
            "input": {
                "encoder_seq": seq_encoder,
                "encoder_pos": pos_encoder,
                "encoder_residue_features": residue_features_encoder,
                "encoder_entity_tags": entity_tags_encoder,
                "encoder_t_embed": t_embed_encoder,
                "seq": seq_docoder,
                "pos": pos_decoder,
                "residue_features": residue_features_decoder,
                "entity_tags": entity_tags_decoder,
            },
            "target": {
                "seq": parse_pose_result.seq[parse_pose_result.cdr_flags == 1],
                "coords": parse_pose_result.coords[parse_pose_result.cdr_flags == 1],
                "left_anchor": parse_pose_result.coords[parse_pose_result.anchor_flags == 1][0],
            },
        }

    def collate_fn(self, batch: List[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
        safe_batch = [sample for sample in batch if sample is not None]
        if len(safe_batch) == 0:
            logging.warning("Empty batch")
            return None

        max_len = max([s["input"]["seq"].shape[1] for s in safe_batch])
        max_len_encoder = max([s["input"]["encoder_seq"].shape[1] for s in safe_batch])
        num_atoms_residue_features = safe_batch[0]["input"]["residue_features"].shape[2]

        encoder_seq = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len_encoder, dtype=torch.int64)
        encoder_pos = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len_encoder, 3)
        encoder_residue_features = torch.zeros(
            len(safe_batch) * self.samples_in_batch, max_len_encoder, num_atoms_residue_features, 3
        )
        encoder_entity_tags = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len_encoder, dtype=torch.int64)
        encoder_t_embed = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len_encoder, dtype=torch.int64)
        seq = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len, dtype=torch.int64)
        pos = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len, 3)
        residue_features = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len, num_atoms_residue_features, 3)
        entity_tags = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len, dtype=torch.int64)
        target_seq = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len, dtype=torch.int64)
        target_coords = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len, num_atoms_residue_features + 1, 3)
        encoder_mask = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len_encoder)
        decoder_mask = torch.zeros(len(safe_batch) * self.samples_in_batch, max_len)
        target_anchor = torch.zeros(len(safe_batch) * self.samples_in_batch, num_atoms_residue_features + 1, 3)

        for i, s in enumerate(safe_batch):
            encoder_seq_len = s["input"]["encoder_seq"].shape[1]
            seq_len = s["input"]["seq"].shape[1]
            encoder_seq[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :encoder_seq_len] = s["input"][
                "encoder_seq"
            ]
            encoder_pos[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :encoder_seq_len] = s["input"][
                "encoder_pos"
            ]
            encoder_residue_features[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :encoder_seq_len] = s[
                "input"
            ]["encoder_residue_features"]
            encoder_entity_tags[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :encoder_seq_len] = s[
                "input"
            ]["encoder_entity_tags"]
            encoder_t_embed[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :encoder_seq_len] = s["input"][
                "encoder_t_embed"
            ]
            seq[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :seq_len] = s["input"]["seq"]
            pos[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :seq_len] = s["input"]["pos"]
            residue_features[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :seq_len] = s["input"][
                "residue_features"
            ]
            entity_tags[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :seq_len] = s["input"][
                "entity_tags"
            ]
            target_seq[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :seq_len] = s["target"][
                "seq"
            ].repeat(self.samples_in_batch, 1)
            target_coords[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :seq_len] = s["target"][
                "coords"
            ].repeat(self.samples_in_batch, 1, 1, 1)
            encoder_mask[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :encoder_seq_len] = 1
            decoder_mask[i * self.samples_in_batch : (i + 1) * self.samples_in_batch, :seq_len] = 1
            target_anchor[i * self.samples_in_batch : (i + 1) * self.samples_in_batch] = s["target"][
                "left_anchor"
            ].repeat(self.samples_in_batch, 1, 1)

        minibatch = torch.arange(len(safe_batch) * self.samples_in_batch).repeat_interleave(max_len, dim=0)
        seq = seq.reshape(-1)
        pos = pos.reshape((-1,) + pos.shape[2:])
        residue_features = residue_features.reshape((-1,) + residue_features.shape[2:])
        target_seq = target_seq.reshape(-1)
        target_coords = target_coords.reshape((-1,) + target_coords.shape[2:])
        encoder_pos_input = torch.cat([encoder_pos.unsqueeze(2), encoder_residue_features], dim=2)

        return {
            "input": {
                "encoder_seq_input": encoder_seq,  # (B, S)
                "encoder_pos_input": encoder_pos_input,  # (B, S, N+1, 3)
                "encoder_entity_tags": encoder_entity_tags,  # (B, S)
                "encoder_t_embed": encoder_t_embed,  # (B, S)
                "encoder_mask": encoder_mask,  # (B, S)
                "seq": seq,  # (B * T)
                "pos": pos,  # (B * T, 3)
                "residue_features": residue_features,  # (B * T, N, 3)
                "entity_tags": entity_tags,  # (B, T)
                "decoder_mask": decoder_mask,  # (B, T)
                "batch": minibatch,  # (B * T)
                "generate_mask": decoder_mask.reshape(-1),  # (B * T)
            },
            "target": {
                "seq": target_seq,  # (B * T)
                "coords": target_coords,  # (B * T, N + 1, 3)
                "left_anchor": target_anchor,
                "generate_mask": decoder_mask.reshape(-1),  # (B * T)
                "batch": minibatch,  # (B * T)
            },
        }
