import os
import re
from typing import List, Sequence

import torch
from PIL import Image

try:
    from data.personal_dataset import find_sample_image_path, QWENVL_IMAGE_TOKEN
except ModuleNotFoundError:
    from ..data.personal_dataset import find_sample_image_path, QWENVL_IMAGE_TOKEN


def _stack_float_tensors(values):
    return torch.tensor(values, dtype=torch.bfloat16)


def _move_image_to_list(image_path: str):
    if not image_path:
        return None
    if not os.path.exists(image_path):
        return None
    with Image.open(image_path) as image:
        return image.convert("RGB")


def ensure_qwenvl_image_prompt(prompt: str, query_mode: str) -> str:
    if query_mode != "image" or QWENVL_IMAGE_TOKEN in prompt:
        return prompt
    replacement = f"[Target Image]: {QWENVL_IMAGE_TOKEN}\n"
    updated_prompt, count = re.subn(
        r"\[Target Image\]:.*?\n",
        replacement,
        prompt,
        count=1,
        flags=re.DOTALL,
    )
    if count:
        return updated_prompt
    return prompt + "\n" + replacement


class QwenVLDEPDataCollator:
    def __init__(
        self,
        processor,
        dataset_root: str,
        dataset_name: str,
        max_length: int,
        training: bool = True,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.training = training

    def _build_prompt_text(self, feature):
        return ensure_qwenvl_image_prompt(
            feature["inp_str"],
            feature.get("query_mode", ""),
        )

    def _build_full_text(self, feature):
        text = self._build_prompt_text(feature)
        if self.training:
            text = f"{text}\n{feature['out_str']}{self.tokenizer.eos_token}"
        return text

    def _collect_images(self, features: Sequence[dict]):
        images = []
        for feature in features:
            prompt = self._build_prompt_text(feature)
            image_path = feature.get("image_path", "")
            if not image_path and feature.get("query_mode") == "image":
                image_path = find_sample_image_path(
                    self.dataset_root,
                    self.dataset_name,
                    feature.get("sample_id", ""),
                )
            if QWENVL_IMAGE_TOKEN in prompt and not image_path:
                raise FileNotFoundError(
                    "Image query requires a real image file, but no image_path was found "
                    f"for sample_id={feature.get('sample_id', '')}."
                )
            image = _move_image_to_list(image_path)
            if image is not None and QWENVL_IMAGE_TOKEN in prompt:
                images.append(image)
        return images

    def _build_target_lengths(self, features):
        if not self.training:
            return [0 for _ in features]
        lengths = []
        for feature in features:
            target_ids = self.tokenizer(
                feature["out_str"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            lengths.append(min(len(target_ids) + 1, self.max_length))
        return lengths

    def __call__(self, features: List[dict]):
        prompt_texts = [self._build_prompt_text(feature) for feature in features]
        full_texts = [self._build_full_text(feature) for feature in features]
        images = self._collect_images(features)

        batch = self.processor(
            text=full_texts,
            images=images if images else None,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        target_lengths = self._build_target_lengths(features)

        labels = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]
        for idx in range(labels.size(0)):
            seq_len = int(attention_mask[idx].sum().item())
            target_len = min(int(target_lengths[idx]), seq_len)
            left_pad = labels.size(1) - seq_len
            labels[idx, : labels.size(1) - target_len] = -100
            labels[idx, attention_mask[idx] == 0] = -100
        batch["labels"] = labels

        if "his_diff_emb" in features[0]:
            batch["his_diff_emb"] = _stack_float_tensors(
                [feature["his_diff_emb"] for feature in features]
            )
        if "label_attn_target_emb" in features[0]:
            batch["label_attn_target_emb"] = _stack_float_tensors(
                [feature["label_attn_target_emb"] for feature in features]
            )
        if "label_attn_retrieved_emb" in features[0]:
            batch["label_attn_retrieved_emb"] = _stack_float_tensors(
                [feature["label_attn_retrieved_emb"] for feature in features]
            )
        if "label_attn_label_mask" in features[0]:
            batch["label_attn_label_mask"] = _stack_float_tensors(
                [feature["label_attn_label_mask"] for feature in features]
            )
        if "label_attn_valid_mask" in features[0]:
            batch["label_attn_valid_mask"] = _stack_float_tensors(
                [feature["label_attn_valid_mask"] for feature in features]
            )
        if "target_label_mask" in features[0]:
            batch["target_label_mask"] = _stack_float_tensors(
                [feature["target_label_mask"] for feature in features]
            )
        return batch


def build_qwenvl_generation_batch(processor, features: List[dict], max_length: int):
    prompt_texts = [
        ensure_qwenvl_image_prompt(feature["inp_str"], feature.get("query_mode", ""))
        for feature in features
    ]
    images = []
    for feature, prompt in zip(features, prompt_texts):
        image = _move_image_to_list(feature.get("image_path", ""))
        if image is None and QWENVL_IMAGE_TOKEN in prompt:
            raise FileNotFoundError(
                "Image query requires a real image file, but no image_path was found "
                f"for sample_id={feature.get('sample_id', '')}."
            )
        if image is not None and QWENVL_IMAGE_TOKEN in prompt:
            images.append(image)
    batch = processor(
        text=prompt_texts,
        images=images if images else None,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return batch
