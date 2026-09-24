import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

try:
    from transformers import Qwen2VLForConditionalGeneration
except Exception as exc:
    Qwen2VLForConditionalGeneration = None
    _QWENVL_IMPORT_ERROR = exc
else:
    _QWENVL_IMPORT_ERROR = None

DEFAULT_EMBED_SIZE = 1024
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_PROMPT_TOKEN_COUNT = 3
DEFAULT_LABEL_TOKEN_COUNT = 0
DEFAULT_LABEL_ATTN_SIZE = 256


def resolve_llm_hidden_size(config, fallback_model: Optional[nn.Module] = None) -> int:
    candidate_sizes = [
        getattr(getattr(config, "text_config", None), "hidden_size", None),
        getattr(config, "hidden_size", None),
    ]
    if fallback_model is not None:
        candidate_sizes.extend(
            [
                getattr(getattr(fallback_model, "config", None), "hidden_size", None),
                getattr(getattr(getattr(fallback_model, "config", None), "text_config", None), "hidden_size", None),
            ]
        )
    for candidate in candidate_sizes:
        if candidate is not None:
            return int(candidate)
    raise ValueError(
        "Could not infer the language model hidden size from the provided Qwen config."
    )


class _MissingQwen2VLForConditionalGeneration(nn.Module):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Qwen2VLForConditionalGeneration could not be imported. "
            "Please run this project in an environment with a Qwen2-VL compatible "
            "transformers/torch installation."
        ) from _QWENVL_IMPORT_ERROR

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError(
            "Qwen2VLForConditionalGeneration could not be imported. "
            "Please run this project in an environment with a Qwen2-VL compatible "
            "transformers/torch installation."
        ) from _QWENVL_IMPORT_ERROR


QwenVLBaseModel = (
    Qwen2VLForConditionalGeneration
    if Qwen2VLForConditionalGeneration is not None
    else _MissingQwen2VLForConditionalGeneration
)


def move_or_materialize_module(module: nn.Module, device: torch.device) -> nn.Module:
    has_meta_param = any(param.is_meta for param in module.parameters(recurse=True))
    if has_meta_param:
        module.to_empty(device=device)
        for submodule in module.modules():
            if hasattr(submodule, "reset_parameters"):
                submodule.reset_parameters()
        return module
    return module.to(device=device)


class SparseAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size, dtype=torch.bfloat16),
            nn.GELU(),
        )
        self.rho = 0.05
        self.last_z = None

    def forward(self, x):
        z = self.encoder(x)
        self.last_z = z
        x_recon = self.decoder(z)
        return z, x_recon
    
    def sae_loss(self, x, x_recon):
        recon_loss = F.smooth_l1_loss(x, x_recon)
        sparse_loss = self.last_z.abs().mean()
        return recon_loss, sparse_loss


class LabelAttentionPrompt(nn.Module):
    def __init__(self, input_size, attn_size, label_count, llm_hidden_size):
        super().__init__()
        self.label_count = label_count
        self.query = nn.Linear(input_size, attn_size, dtype=torch.bfloat16)
        self.key = nn.Linear(input_size, attn_size, dtype=torch.bfloat16)
        self.label_embedding = nn.Embedding(label_count, attn_size, dtype=torch.bfloat16)
        self.prompt_mlp = nn.Sequential(
            nn.Linear(attn_size + 1, llm_hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size, dtype=torch.bfloat16),
        )
        self.temperature = nn.Parameter(torch.tensor(0.07, dtype=torch.bfloat16))
        self.gate = nn.Parameter(torch.tensor(-5.0, dtype=torch.bfloat16))

    def forward(self, target_emb, retrieved_emb, retrieved_label_mask, valid_mask):
        target_emb = target_emb.to(dtype=torch.bfloat16)
        retrieved_emb = retrieved_emb.to(dtype=torch.bfloat16)
        retrieved_label_mask = retrieved_label_mask.to(dtype=torch.bfloat16)
        valid_mask = valid_mask.to(dtype=torch.bfloat16)

        query = self.query(target_emb).unsqueeze(1)
        key = self.key(retrieved_emb)
        cosine_scores = F.cosine_similarity(query, key, dim=-1)
        temperature = self.temperature.float().abs().clamp_min(1e-4)
        masked_scores = cosine_scores.float().masked_fill(valid_mask <= 0, -1e4)
        instance_scores = F.softmax(masked_scores / temperature, dim=-1).to(dtype=torch.bfloat16)
        instance_scores = instance_scores.masked_fill(valid_mask <= 0, 0.0)

        label_membership = retrieved_label_mask * valid_mask.unsqueeze(-1)
        label_counts = label_membership.sum(dim=1).clamp_min(1.0)
        label_scores = (
            label_membership * instance_scores.unsqueeze(-1)
        ).sum(dim=1) / label_counts

        label_ids = torch.arange(
            self.label_count,
            device=target_emb.device,
            dtype=torch.long,
        )
        label_emb = self.label_embedding(label_ids).unsqueeze(0).expand(
            target_emb.size(0),
            -1,
            -1,
        )
        prompt_input = torch.cat([label_emb, label_scores.unsqueeze(-1)], dim=-1)
        label_prompt_emb = self.prompt_mlp(prompt_input)
        gate = torch.sigmoid(self.gate.float()).to(label_prompt_emb.dtype)
        return label_prompt_emb, label_scores, instance_scores, gate


class DEPModel(QwenVLBaseModel):

    def __init__(self, config):
        super().__init__(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_hidden_size = resolve_llm_hidden_size(config, getattr(self, "model", None))
        self.rag_embed_size = getattr(config, "rag_embed_size", DEFAULT_EMBED_SIZE)
        self.sparse_hidden_size = getattr(config, "sparse_hidden_size", DEFAULT_HIDDEN_SIZE)
        self.prompt_token_count = getattr(config, "prompt_token_count", DEFAULT_PROMPT_TOKEN_COUNT)
        self.label_token_count = getattr(config, "label_token_count", DEFAULT_LABEL_TOKEN_COUNT)
        self.label_attn_size = getattr(config, "label_attn_size", DEFAULT_LABEL_ATTN_SIZE)
        self.label_attn_loss_weight = getattr(config, "label_attn_loss_weight", 0.0)
        self.sae_loss_weight = getattr(config, "sae_loss_weight", 0.1)
        self.sae_sparse_weight = getattr(config, "sae_sparse_weight", 1e-3)
        self.label_loss_weight = getattr(config, "label_loss_weight", 0.8)
        self._explanation_pattern_ids = None

        self.sae = SparseAutoEncoder(self.rag_embed_size, self.sparse_hidden_size)
        move_or_materialize_module(self.sae, device)
        self.align_mlp_his = nn.Sequential(
            nn.Linear(self.sparse_hidden_size, self.llm_hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size, dtype=torch.bfloat16),
        )
        move_or_materialize_module(self.align_mlp_his, device)
        self.align_mlp_diff = nn.Sequential(
            nn.Linear(self.sparse_hidden_size, self.llm_hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size, dtype=torch.bfloat16),
        )
        move_or_materialize_module(self.align_mlp_diff, device)
        if self.label_token_count > 0:
            self.label_attention = LabelAttentionPrompt(
                self.rag_embed_size,
                self.label_attn_size,
                self.label_token_count,
                self.llm_hidden_size,
            )
            move_or_materialize_module(self.label_attention, device)
        else:
            self.label_attention = None

    def _build_sae_input(self, his_diff_emb: torch.Tensor) -> torch.Tensor:
        if his_diff_emb is None:
            raise ValueError("`his_diff_emb` is required for DEPModel.")
        if his_diff_emb.dim() != 3:
            raise ValueError(
                f"`his_diff_emb` must be a 3D tensor, but got shape {tuple(his_diff_emb.shape)}."
            )
        if his_diff_emb.size(-1) != self.rag_embed_size:
            raise ValueError(
                "The last dimension of `his_diff_emb` does not match the configured "
                f"retrieval embedding size: got {his_diff_emb.size(-1)}, expected {self.rag_embed_size}."
            )

        # Backward-compatible path: the dataset directly provides [self; diff] prompts.
        if his_diff_emb.size(1) == self.prompt_token_count * 2:
            return his_diff_emb

        # New path: first vector is the target query embedding, followed by retrieved triplet embeddings.
        if his_diff_emb.size(1) == self.prompt_token_count + 1:
            target_emb = his_diff_emb[:, :1, :]
            retrieved_emb = his_diff_emb[:, 1:, :]
            self_emb = target_emb.expand(-1, self.prompt_token_count, -1)
            diff_emb = target_emb - retrieved_emb
            return torch.cat([self_emb, diff_emb], dim=1)

        raise ValueError(
            "`his_diff_emb` must contain either 2 * prompt_token_count vectors "
            "(legacy DEP format) or prompt_token_count + 1 vectors "
            "(target + retrieved RAG format). "
            f"Received shape: {tuple(his_diff_emb.shape)}."
        )

    def _replace_soft_prompt_tokens(
        self,
        inputs_embs: torch.Tensor,
        input_ids: torch.Tensor,
        prompt_emb: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        new_tokens = [f"[{prefix}_TOKEN_{i}]" for i in range(self.prompt_token_count)]
        vocab = self.llm_tokenizer.get_vocab()
        token_ids = [vocab.get(token) for token in new_tokens]
        prompt_emb = prompt_emb.to(inputs_embs.dtype)
        updated_embs = inputs_embs
        input_ids = input_ids.to(inputs_embs.device)

        for i, token_id in enumerate(token_ids):
            if token_id is None or token_id < 0:
                continue
            token_mask = (input_ids == token_id).unsqueeze(-1)
            replacement = prompt_emb[:, i, :].unsqueeze(1).to(
                device=inputs_embs.device,
                dtype=inputs_embs.dtype,
            )
            updated_embs = torch.where(token_mask, replacement, updated_embs)

        return updated_embs

    def _replace_label_attention_tokens(
        self,
        inputs_embs: torch.Tensor,
        input_ids: torch.Tensor,
        label_prompt_emb: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if self.label_token_count <= 0:
            return inputs_embs
        new_tokens = [f"[LAB_ATTN_{i}]" for i in range(self.label_token_count)]
        vocab = self.llm_tokenizer.get_vocab()
        token_ids = [vocab.get(token) for token in new_tokens]
        label_prompt_emb = label_prompt_emb.to(inputs_embs.dtype)
        updated_embs = inputs_embs
        input_ids = input_ids.to(inputs_embs.device)
        gate = gate.to(device=inputs_embs.device, dtype=inputs_embs.dtype)

        for i, token_id in enumerate(token_ids):
            if token_id is None or token_id < 0:
                continue
            token_mask = (input_ids == token_id).unsqueeze(-1).to(inputs_embs.dtype)
            addition = gate * label_prompt_emb[:, i, :].unsqueeze(1).to(
                device=inputs_embs.device,
                dtype=inputs_embs.dtype,
            )
            updated_embs = updated_embs + token_mask * addition

        return updated_embs

    def _compute_label_attention_loss(self, label_scores, target_label_mask):
        if target_label_mask is None or self.label_token_count <= 0:
            return None
        target_label_mask = target_label_mask.to(
            device=label_scores.device,
            dtype=label_scores.dtype,
        )
        if target_label_mask.shape != label_scores.shape:
            return None
        return F.binary_cross_entropy_with_logits(
            label_scores.float(),
            target_label_mask.float(),
        )

    def freeze_non_dep_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = (
                "align_mlp" in name
                or "sae" in name
                or "label_attention" in name
            )

    def _get_explanation_pattern_ids(self):
        if self._explanation_pattern_ids is not None:
            return self._explanation_pattern_ids
        if not hasattr(self, "llm_tokenizer") or self.llm_tokenizer is None:
            return []
        self._explanation_pattern_ids = self.llm_tokenizer(
            "Explanation:",
            add_special_tokens=False,
        )["input_ids"]
        return self._explanation_pattern_ids

    @staticmethod
    def _find_subsequence_start(sequence, pattern):
        if not pattern or len(sequence) < len(pattern):
            return -1
        pattern_len = len(pattern)
        for idx in range(len(sequence) - pattern_len + 1):
            if sequence[idx: idx + pattern_len] == pattern:
                return idx
        return -1

    def _split_label_explanation_loss(self, logits, labels, **kwargs):
        if logits.size(1) != labels.size(1):
            return self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=logits.size(-1),
                **kwargs,
            )

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).float(),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        token_loss = flat_token_loss.view_as(shift_labels)
        valid_mask = shift_labels.ne(-100)
        label_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        explanation_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        explanation_pattern = self._get_explanation_pattern_ids()

        for bidx in range(shift_labels.size(0)):
            valid_positions = torch.nonzero(valid_mask[bidx], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            target_tokens = shift_labels[bidx, valid_positions].tolist()
            explanation_start = self._find_subsequence_start(target_tokens, explanation_pattern)
            if explanation_start < 0:
                label_mask[bidx, valid_positions] = True
                explanation_mask[bidx, valid_positions] = True
                continue

            label_positions = valid_positions[:explanation_start]
            explanation_positions = valid_positions[explanation_start:]
            if label_positions.numel() == 0:
                label_positions = valid_positions
            if explanation_positions.numel() == 0:
                explanation_positions = valid_positions
            label_mask[bidx, label_positions] = True
            explanation_mask[bidx, explanation_positions] = True

        if not label_mask.any() or not explanation_mask.any():
            return token_loss[valid_mask].mean()

        label_loss = token_loss[label_mask].mean()
        explanation_loss = token_loss[explanation_mask].mean()
        label_weight = min(max(float(self.label_loss_weight), 0.0), 1.0)
        return label_weight * label_loss + (1.0 - label_weight) * explanation_loss

    def _build_label_explanation_masks(self, shift_labels):
        valid_mask = shift_labels.ne(-100)
        label_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        explanation_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        explanation_pattern = self._get_explanation_pattern_ids()

        for bidx in range(shift_labels.size(0)):
            valid_positions = torch.nonzero(valid_mask[bidx], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            target_tokens = shift_labels[bidx, valid_positions].tolist()
            explanation_start = self._find_subsequence_start(target_tokens, explanation_pattern)
            if explanation_start < 0:
                label_mask[bidx, valid_positions] = True
                explanation_mask[bidx, valid_positions] = True
                continue

            label_positions = valid_positions[:explanation_start]
            explanation_positions = valid_positions[explanation_start:]
            if label_positions.numel() == 0:
                label_positions = valid_positions
            if explanation_positions.numel() == 0:
                explanation_positions = valid_positions
            label_mask[bidx, label_positions] = True
            explanation_mask[bidx, explanation_positions] = True

        return valid_mask, label_mask, explanation_mask

    def _token_loss_from_hidden(self, shift_hidden, shift_labels, mask):
        if not mask.any():
            return None
        selected_hidden = shift_hidden[mask]
        selected_labels = shift_labels[mask]
        logits = self.lm_head(selected_hidden)
        return F.cross_entropy(
            logits.float(),
            selected_labels,
            reduction="mean",
        )

    def _compute_label_explanation_loss_from_hidden(self, hidden_states, labels):
        if hidden_states.size(1) != labels.size(1):
            logits = self.lm_head(hidden_states)
            return self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=logits.size(-1),
            )

        shift_hidden = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask, label_mask, explanation_mask = self._build_label_explanation_masks(shift_labels)
        if not valid_mask.any():
            return shift_hidden.sum() * 0.0

        supervised_mask = label_mask | explanation_mask
        if not supervised_mask.any():
            supervised_mask = valid_mask

        selected_hidden = shift_hidden[supervised_mask]
        selected_labels = shift_labels[supervised_mask]
        selected_logits = self.lm_head(selected_hidden)
        selected_loss = F.cross_entropy(
            selected_logits.float(),
            selected_labels,
            reduction="none",
        )
        token_loss = shift_hidden.new_zeros(shift_labels.shape, dtype=torch.float32)
        token_loss[supervised_mask] = selected_loss

        if not label_mask.any() or not explanation_mask.any():
            return token_loss[supervised_mask].mean()

        label_loss = token_loss[label_mask].mean()
        explanation_loss = token_loss[explanation_mask].mean()
        label_weight = min(max(float(self.label_loss_weight), 0.0), 1.0)
        return label_weight * label_loss + (1.0 - label_weight) * explanation_loss

    def _build_multimodal_inputs_embeds(
        self,
        input_ids,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        if inputs_embeds is not None:
            return inputs_embeds
        if input_ids is None:
            raise ValueError("`input_ids` is required when `inputs_embeds` is not provided.")

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` is required when `pixel_values` is provided.")
            image_embeds = self.model.get_image_features(
                pixel_values,
                image_grid_thw,
            ).pooler_output
            if isinstance(image_embeds, (tuple, list)):
                image_embeds = torch.cat(image_embeds, dim=0)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens: {n_image_tokens}, features: {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("`video_grid_thw` is required when `pixel_values_videos` is provided.")
            video_embeds = self.model.get_video_features(
                pixel_values_videos,
                video_grid_thw,
            ).pooler_output
            if isinstance(video_embeds, (tuple, list)):
                video_embeds = torch.cat(video_embeds, dim=0)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    "Video features and video tokens do not match: "
                    f"tokens: {n_video_tokens}, features: {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds

    @staticmethod
    def _past_key_values_empty(past_key_values):
        if past_key_values is None:
            return True
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length() == 0
        return False

    def _prepare_multimodal_position_ids(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        image_grid_thw,
        video_grid_thw,
        mm_token_type_ids,
    ):
        if position_ids is not None:
            return position_ids
        if hasattr(self.model, "compute_3d_position_ids"):
            return self.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )
        return position_ids

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        his_diff_emb: Optional[torch.FloatTensor] = None,
        label_attn_target_emb: Optional[torch.FloatTensor] = None,
        label_attn_retrieved_emb: Optional[torch.FloatTensor] = None,
        label_attn_label_mask: Optional[torch.FloatTensor] = None,
        label_attn_valid_mask: Optional[torch.FloatTensor] = None,
        target_label_mask: Optional[torch.FloatTensor] = None,
        use_sae_dsp: bool = True,
        use_label_dsp: bool = True,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = True

        inputs_embs = self._build_multimodal_inputs_embeds(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embs.device)
        sae_input = None
        his_diff_recon_emb = None
        if use_sae_dsp:
            sae_input = self._build_sae_input(his_diff_emb).to(
                device=inputs_embs.device,
                dtype=torch.bfloat16,
            )
            his_diff_sparse_emb, his_diff_recon_emb = self.sae(sae_input)
            his_emb = his_diff_sparse_emb[:, :self.prompt_token_count, :]
            diff_emb = his_diff_sparse_emb[:, self.prompt_token_count:, :]

            his_emb = self.align_mlp_his(his_emb)
            diff_emb = self.align_mlp_diff(diff_emb)
            inputs_embs = self._replace_soft_prompt_tokens(inputs_embs, input_ids, his_emb, "HIS")
            inputs_embs = self._replace_soft_prompt_tokens(inputs_embs, input_ids, diff_emb, "DIFF")
        label_scores = None
        if (
            use_label_dsp
            and self.label_attention is not None
            and label_attn_target_emb is not None
            and label_attn_retrieved_emb is not None
            and label_attn_label_mask is not None
            and label_attn_valid_mask is not None
        ):
            label_prompt_emb, label_scores, _, label_gate = self.label_attention(
                label_attn_target_emb.to(device=inputs_embs.device),
                label_attn_retrieved_emb.to(device=inputs_embs.device),
                label_attn_label_mask.to(device=inputs_embs.device),
                label_attn_valid_mask.to(device=inputs_embs.device),
            )
            inputs_embs = self._replace_label_attention_tokens(
                inputs_embs,
                input_ids,
                label_prompt_emb,
                label_gate,
            )

        position_ids = self._prepare_multimodal_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embs,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embs,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
        )

        hidden_states = outputs.last_hidden_state
        if labels is None:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            logits = self.lm_head(hidden_states[:, -1:, :])

        loss = None
        if labels is not None:
            llm_loss = self._compute_label_explanation_loss_from_hidden(hidden_states, labels)
            loss = llm_loss
            if use_sae_dsp:
                recon_loss, sparse_loss = self.sae.sae_loss(sae_input, his_diff_recon_emb)
                loss = loss + self.sae_loss_weight * (
                    recon_loss + self.sae_sparse_weight * sparse_loss
                )
            if use_label_dsp and label_scores is not None and self.label_attn_loss_weight > 0:
                label_attn_loss = self._compute_label_attention_loss(
                    label_scores,
                    target_label_mask,
                )
                if label_attn_loss is not None:
                    loss = loss + self.label_attn_loss_weight * label_attn_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        *args,
        his_diff_emb=None,
        label_attn_target_emb=None,
        label_attn_retrieved_emb=None,
        label_attn_label_mask=None,
        label_attn_valid_mask=None,
        target_label_mask=None,
        use_sae_dsp=True,
        use_label_dsp=True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if his_diff_emb is not None:
            model_inputs["his_diff_emb"] = his_diff_emb
        if label_attn_target_emb is not None:
            model_inputs["label_attn_target_emb"] = label_attn_target_emb
        if label_attn_retrieved_emb is not None:
            model_inputs["label_attn_retrieved_emb"] = label_attn_retrieved_emb
        if label_attn_label_mask is not None:
            model_inputs["label_attn_label_mask"] = label_attn_label_mask
        if label_attn_valid_mask is not None:
            model_inputs["label_attn_valid_mask"] = label_attn_valid_mask
        if target_label_mask is not None:
            model_inputs["target_label_mask"] = target_label_mask
        model_inputs["use_sae_dsp"] = use_sae_dsp
        model_inputs["use_label_dsp"] = use_label_dsp
        return model_inputs
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        training: bool = False,
        tokenizer: Optional[AutoTokenizer] = None,
        rag_embed_size: Optional[int] = None,
        sparse_hidden_size: Optional[int] = None,
        prompt_token_count: Optional[int] = None,
        label_token_count: Optional[int] = None,
        label_attn_size: Optional[int] = None,
        label_attn_loss_weight: Optional[float] = None,
        sae_loss_weight: Optional[float] = None,
        sae_sparse_weight: Optional[float] = None,
        label_loss_weight: Optional[float] = None,
        **kwargs
    ):
        if config is None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
        if rag_embed_size is not None:
            setattr(config, "rag_embed_size", rag_embed_size)
        if sparse_hidden_size is not None:
            setattr(config, "sparse_hidden_size", sparse_hidden_size)
        if prompt_token_count is not None:
            setattr(config, "prompt_token_count", prompt_token_count)
        if label_token_count is not None:
            setattr(config, "label_token_count", label_token_count)
        if label_attn_size is not None:
            setattr(config, "label_attn_size", label_attn_size)
        if label_attn_loss_weight is not None:
            setattr(config, "label_attn_loss_weight", label_attn_loss_weight)
        if sae_loss_weight is not None:
            setattr(config, "sae_loss_weight", sae_loss_weight)
        if sae_sparse_weight is not None:
            setattr(config, "sae_sparse_weight", sae_sparse_weight)
        if label_loss_weight is not None:
            setattr(config, "label_loss_weight", label_loss_weight)
        setattr(config, "architectures", ["DEPModel"])

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )
        model.llm_tokenizer = tokenizer
        if training:
            model.freeze_non_dep_parameters()
        return model
