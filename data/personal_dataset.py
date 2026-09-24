import json
import os
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from utils.templates import Qwen2PromptTemplate
except ModuleNotFoundError:
    from ..utils.templates import Qwen2PromptTemplate

MMIMDB_LABELS = [
    "Drama",
    "Comedy",
    "Romance",
    "Thriller",
    "Crime",
    "Action",
    "Adventure",
    "Horror",
    "Documentary",
    "Mystery",
    "Sci-Fi",
    "Fantasy",
    "Family",
    "Biography",
    "War",
    "History",
    "Music",
    "Animation",
    "Musical",
    "Western",
    "Sport",
    "Short",
    "Film-Noir",
]

QWENVL_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_EXTENSIONS = ("jpeg", "jpg", "png", "webp")

HATEMEMES_LABELS = {
    0: "non-hateful",
    1: "hateful",
}


def _normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def _truncate_text(text, max_chars):
    text = _normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _to_scalar_label(label):
    if isinstance(label, np.generic):
        return label.item()
    if isinstance(label, np.ndarray):
        if label.ndim == 0:
            return label.item()
        return label.tolist()
    return label


def find_sample_image_path(dataset_root, dataset_name, sample_id):
    image_dir = Path(dataset_root) / dataset_name / "image"
    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{sample_id}.{extension}"
        if image_path.exists():
            return str(image_path)
    return ""


def load_label_space(dataset_name, dataset_root):
    dataset_name = dataset_name.lower()
    if dataset_name == "mmimdb":
        return MMIMDB_LABELS
    if dataset_name == "hatememes":
        return HATEMEMES_LABELS
    if dataset_name == "food101":
        class_idx_path = Path(dataset_root) / "food101" / "class_idx.json"
        if not class_idx_path.exists():
            class_idx_path = Path(dataset_root) / "food101" / "meta_data" / "class_idx.json"
        if not class_idx_path.exists():
            raise FileNotFoundError(
                f"Food101 label mapping not found: {class_idx_path}"
            )
        with open(class_idx_path, "r", encoding="utf-8") as f:
            class_idx = json.load(f)
        id_to_label = {int(v): k.replace("_", " ") for k, v in class_idx.items()}
        return id_to_label
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def stringify_label(label, dataset_name, label_space):
    dataset_name = dataset_name.lower()
    label = _to_scalar_label(label)
    if dataset_name == "mmimdb":
        label_array = np.asarray(label).astype(int).tolist()
        labels = [label_space[idx] for idx, value in enumerate(label_array) if value == 1]
        return ", ".join(labels) if labels else "None"
    if isinstance(label_space, dict):
        label_id = int(label)
        return label_space.get(label_id, str(label_id))
    return str(label)


def get_label_names(label_space):
    if isinstance(label_space, dict):
        return [label_space[idx] for idx in sorted(label_space)]
    return list(label_space)


def label_to_multihot(label, dataset_name, label_space):
    label_count = len(label_space)
    multihot = np.zeros(label_count, dtype=np.float32)
    label = _to_scalar_label(label)
    if dataset_name.lower() == "mmimdb":
        label_array = np.asarray(label).astype(int).tolist()
        for idx, value in enumerate(label_array[:label_count]):
            if int(value) == 1:
                multihot[idx] = 1.0
        return multihot
    label_id = int(label)
    if 0 <= label_id < label_count:
        multihot[label_id] = 1.0
    return multihot


def format_retrieved_label_candidates(retrieved_labels, dataset_name, label_space):
    candidates = []
    seen = set()
    for label in retrieved_labels:
        label_text = stringify_label(label, dataset_name, label_space)
        if not label_text or label_text == "None":
            continue
        if dataset_name == "mmimdb":
            items = [part.strip() for part in label_text.split(",") if part.strip()]
        else:
            items = [label_text]
        for item in items:
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            candidates.append(item)
    return ", ".join(candidates) if candidates else "None"


def build_explanation_text(dataset_name, text, max_chars):
    text = _truncate_text(text, max_chars)
    if dataset_name == "hatememes":
        return f"Meme text evidence: {text}"
    if dataset_name == "mmimdb":
        return f"Plot description evidence: {text}"
    if dataset_name == "food101":
        return f"Dish description evidence: {text}"
    return text


def build_target_output(dataset_name, label_text, text, max_chars):
    explanation = build_explanation_text(dataset_name, text, max_chars)
    return f"Label: {label_text}\nExplanation: {explanation}"


def build_system_prompt(dataset_name, label_space):
    if dataset_name == "mmimdb":
        task_line = "Predict one or more movie genres."
        label_line = ", ".join(label_space)
    elif dataset_name == "hatememes":
        task_line = "Predict whether the meme is hateful or non-hateful."
        label_line = ", ".join(label_space.values())
    elif dataset_name == "food101":
        task_line = "Predict the food category."
        label_line = ", ".join(label_space.values())
    else:
        task_line = "Predict the target label."
        label_line = ""

    return (
        f"You are a retrieval-augmented multimodal classification assistant for the {dataset_name} dataset.\n"
        f"{task_line}\n"
        f"The dynamic soft prompts [Self Prompt] and [Difference Prompt] are generated from the target sample "
        f"and retrieved multimodal memory-bank triplets.\n"
        f"Allowed labels: {label_line}\n"
        "The Label field must contain only labels from the Allowed labels list. "
        "Do not invent, paraphrase, or output labels outside the allowed set.\n"
        "Always answer using exactly the following format:\n"
        "Label: <comma-separated labels from Allowed labels only>\n"
        "Explanation: <concise explanation>"
    )


class MemoryBankCache:
    def __init__(self, dataset_name, memory_root):
        self.dataset_name = dataset_name
        self.memory_root = Path(memory_root)
        self.cache = {}

    def _load_array(self, subdir, item_id):
        path = self.memory_root / self.dataset_name / subdir / f"{item_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Memory bank file not found: {path}")
        return np.load(path).astype(np.float32)

    def get(self, item_id):
        item_id = str(item_id)
        if item_id not in self.cache:
            text_emb = self._load_array("text", item_id)
            image_emb = self._load_array("image", item_id)
            text_pool = text_emb.mean(axis=0)
            image_pool = image_emb.mean(axis=0)
            fused_pool = (text_pool + image_pool) / 2.0
            self.cache[item_id] = {
                "text": text_pool.astype(np.float32),
                "image": image_pool.astype(np.float32),
                "fused": fused_pool.astype(np.float32),
            }
        return self.cache[item_id]


class PersonalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        llm_tokenizer,
        memory_root="dataset/memory_bank",
        dataset_root="dataset",
        max_length=2048,
        max_target_length=256,
        prompt_token_count=3,
        new_tokens=None,
        query_modes=("text", "image"),
        explanation_max_chars=256,
        training=True,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_name = dataset_name.lower()
        self.llm_tokenizer = llm_tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.prompt_token_count = prompt_token_count
        self.query_modes = tuple(mode for mode in query_modes if mode in {"text", "image"})
        self.training = training
        self.dataset_root = Path(dataset_root)
        self.label_space = load_label_space(self.dataset_name, dataset_root)
        self.label_names = get_label_names(self.label_space)
        self.label_token_count = len(self.label_names)
        self.memory_bank = MemoryBankCache(self.dataset_name, memory_root)
        self.processed_data = []

        if not self.query_modes:
            raise ValueError("At least one query mode must be enabled: text and/or image.")

        if new_tokens is None:
            new_tokens = [f"[HIS_TOKEN_{i}]" for i in range(prompt_token_count)] + [
                f"[DIFF_TOKEN_{i}]" for i in range(prompt_token_count)
            ] + [
                f"[LAB_ATTN_{i}]" for i in range(self.label_token_count)
            ] + [
                "<his_token_start>",
                "<his_token_end>",
                "<diff_token_start>",
                "<diff_token_end>",
                "<label_attn_start>",
                "<label_attn_end>",
            ]
        self.new_tokens = new_tokens

        self.self_prompt_tokens = " ".join(f"[HIS_TOKEN_{i}]" for i in range(prompt_token_count))
        self.diff_prompt_tokens = " ".join(f"[DIFF_TOKEN_{i}]" for i in range(prompt_token_count))
        self.pt = Qwen2PromptTemplate(build_system_prompt(self.dataset_name, self.label_space))

        for _, row in tqdm(
            self.dataframe.iterrows(),
            total=len(self.dataframe),
            desc=f"Pre-Processing {self.dataset_name} data",
        ):
            self._process_row(row, explanation_max_chars)

        self.total_len = len(self.processed_data)

    def _build_retrieval_context(self, row, query_mode):
        bank_entry = self.memory_bank.get(row["item_id"])
        target_emb = bank_entry["text"] if query_mode == "text" else bank_entry["image"]

        if query_mode == "text":
            retrieval_ids = list(row.get("t2t_id_list", []))
            retrieval_sims = list(row.get("t2t_sims_list", []))
            retrieval_labels = list(row.get("t2t_label_list", []))
        else:
            retrieval_ids = list(row.get("i2i_id_list", []))
            retrieval_sims = list(row.get("i2i_sims_list", []))
            retrieval_labels = list(row.get("i2i_label_list", []))

        retrieval_ids = retrieval_ids[: self.prompt_token_count]
        retrieval_labels = retrieval_labels[: self.prompt_token_count]
        retrieval_sims = retrieval_sims[: self.prompt_token_count]

        sim_weights = np.asarray(retrieval_sims, dtype=np.float32)
        if sim_weights.size > 0:
            sim_weights = np.maximum(sim_weights, 0.0)
            sim_sum = float(sim_weights.sum())
            if sim_sum > 0:
                sim_weights = sim_weights / sim_sum
            else:
                sim_weights = np.full_like(sim_weights, 1.0 / sim_weights.size)

        retrieved_embs = []
        label_attn_retrieved_embs = []
        label_attn_label_masks = []
        label_attn_valid_mask = []
        for idx in range(self.prompt_token_count):
            if idx < len(retrieval_ids):
                raw_fused_emb = self.memory_bank.get(retrieval_ids[idx])["fused"].copy()
                fused_emb = raw_fused_emb.copy()
                if idx < len(sim_weights):
                    fused_emb = fused_emb * sim_weights[idx]
                retrieved_embs.append(fused_emb.astype(np.float32))
                label_attn_retrieved_embs.append(raw_fused_emb.astype(np.float32))
                label_attn_label_masks.append(
                    label_to_multihot(
                        retrieval_labels[idx],
                        self.dataset_name,
                        self.label_space,
                    )
                )
                label_attn_valid_mask.append(1.0)
            else:
                retrieved_embs.append(np.zeros_like(target_emb, dtype=np.float32))
                label_attn_retrieved_embs.append(np.zeros_like(target_emb, dtype=np.float32))
                label_attn_label_masks.append(
                    np.zeros(self.label_token_count, dtype=np.float32)
                )
                label_attn_valid_mask.append(0.0)

        context_emb = np.stack([target_emb.astype(np.float32)] + retrieved_embs, axis=0)
        return {
            "context_emb": context_emb,
            "retrieval_labels": retrieval_labels,
            "target_emb": target_emb.astype(np.float32),
            "retrieved_emb": np.stack(label_attn_retrieved_embs, axis=0).astype(np.float32),
            "retrieved_label_mask": np.stack(label_attn_label_masks, axis=0).astype(np.float32),
            "valid_mask": np.asarray(label_attn_valid_mask, dtype=np.float32),
        }

    def _build_label_attention_table_prompt(self):
        return self._build_label_attention_table_prompt_for_indices(range(self.label_token_count))

    def _build_label_attention_table_prompt_for_indices(self, label_indices):
        rows = [
            "[Label Attention Table]",
            "| Label | Attention |",
            "| --- | --- |",
        ]
        label_indices = list(label_indices)
        if not label_indices:
            rows.append("| None | No retrieved label candidates |")
        for idx in label_indices:
            label_name = self.label_names[idx]
            rows.append(f"| {label_name} | [LAB_ATTN_{idx}] |")
        return (
            "<label_attn_start>\n"
            + "\n".join(rows)
            + "\n<label_attn_end>\n"
        )

    def _build_candidate_label_indices(self, retrieval_labels):
        seen = set()
        indices = []
        for retrieval_label in retrieval_labels:
            label_mask = label_to_multihot(
                retrieval_label,
                self.dataset_name,
                self.label_space,
            )
            for idx in np.flatnonzero(label_mask > 0):
                idx = int(idx)
                if idx not in seen:
                    seen.add(idx)
                    indices.append(idx)
        return indices

    def _build_input_prompt(self, row, query_mode, candidate_labels, candidate_label_indices):
        if query_mode == "text":
            target_observation = f"[Target Text]: {_normalize_text(row['text'])}\n"
            modality_line = "[Available Modality]: text only\n"
            retrieval_source_line = (
                "[Text Retrieved Label Candidates]: "
                f"{candidate_labels}\n"
            )
        else:
            target_observation = f"[Target Image]: {QWENVL_IMAGE_TOKEN}\n"
            modality_line = "[Available Modality]: image only\n"
            retrieval_source_line = (
                "[Image Retrieved Label Candidates] (higher-confidence): "
                f"{candidate_labels}\n"
            )
        reliability_note = (
            "Reliability note: image retrieval candidates are empirically more reliable on mmimdb; "
            "use retrieved labels as candidates, not as guaranteed answers.\n"
            if self.dataset_name == "mmimdb"
            else "Reliability note: use retrieved labels as candidates, not as guaranteed answers.\n"
        )

        return self.pt.build_prompt(
            f"[Dataset]: {self.dataset_name}\n"
            f"{modality_line}"
            f"[Active Retrieval Query]: {query_mode}\n"
            f"{reliability_note}"
            f"{retrieval_source_line}"
            f"{target_observation}"
            f"[Self Prompt]: <his_token_start>{self.self_prompt_tokens}<his_token_end>\n"
            f"[Difference Prompt]: <diff_token_start>{self.diff_prompt_tokens}<diff_token_end>\n"
            f"{self._build_label_attention_table_prompt_for_indices(candidate_label_indices)}"
            "Predict the label using only the allowed label names and provide a concise explanation. "
            "Do not output any label outside the allowed label list."
        )

    def _tokenize_training_example(self, inp_str, out_str):
        target_ids = self.llm_tokenizer(
            out_str,
            max_length=self.max_target_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        target_ids = target_ids + [self.llm_tokenizer.eos_token_id]
        max_prompt_len = max(1, self.max_length - len(target_ids))
        prompt_ids = self.llm_tokenizer(
            inp_str,
            max_length=max_prompt_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = [self.llm_tokenizer.pad_token_id] * pad_len + input_ids
            attention_mask = [0] * pad_len + attention_mask
            labels = [-100] * pad_len + labels
        else:
            input_ids = input_ids[-self.max_length :]
            attention_mask = attention_mask[-self.max_length :]
            labels = labels[-self.max_length :]

        return {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
            "attention_mask": np.asarray(attention_mask, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
        }

    def _tokenize_infer_example(self, inp_str):
        input_ids = self.llm_tokenizer(
            inp_str,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = [self.llm_tokenizer.pad_token_id] * pad_len + input_ids
        else:
            input_ids = input_ids[-self.max_length :]
        return {
            "input_ids": np.asarray(input_ids, dtype=np.int64),
        }

    def _process_row(self, row, explanation_max_chars):
        label_text = stringify_label(row["label"], self.dataset_name, self.label_space)
        out_str = build_target_output(
            self.dataset_name,
            label_text,
            row["text"],
            explanation_max_chars,
        )

        for query_mode in self.query_modes:
            image_path = ""
            if query_mode == "image":
                image_path = find_sample_image_path(
                    self.dataset_root,
                    self.dataset_name,
                    row["item_id"],
                )
            retrieval_context = self._build_retrieval_context(row, query_mode)
            retrieval_labels = retrieval_context["retrieval_labels"]
            candidate_labels = format_retrieved_label_candidates(
                retrieval_labels,
                self.dataset_name,
                self.label_space,
            )
            candidate_label_indices = self._build_candidate_label_indices(retrieval_labels)
            inp_str = self._build_input_prompt(
                row,
                query_mode,
                candidate_labels,
                candidate_label_indices,
            )
            target_label_mask = label_to_multihot(
                row["label"],
                self.dataset_name,
                self.label_space,
            )

            data = {
                "sample_id": str(row["item_id"]),
                "query_mode": query_mode,
                "image_path": image_path,
                "image_path_exists": int(bool(image_path)),
                "inp_str": inp_str,
                "out_str": out_str,
                "label_text": label_text,
                "his_diff_emb": retrieval_context["context_emb"].astype(np.float32),
                "label_attn_target_emb": retrieval_context["target_emb"].astype(np.float32),
                "label_attn_retrieved_emb": retrieval_context["retrieved_emb"].astype(np.float32),
                "label_attn_label_mask": retrieval_context["retrieved_label_mask"].astype(np.float32),
                "label_attn_valid_mask": retrieval_context["valid_mask"].astype(np.float32),
                "target_label_mask": target_label_mask.astype(np.float32),
            }
            if self.training:
                data.update(self._tokenize_training_example(inp_str, out_str))
            else:
                data.update(self._tokenize_infer_example(inp_str))
            self.processed_data.append(data)

    def __len__(self):
        return self.total_len

    def get_output(self, idx):
        return self.processed_data[idx]["out_str"]

    def __getitem__(self, idx):
        return self.processed_data[idx]


def convert_to_dataset(dataset):
    def gen():
        for data in dataset:
            yield data

    return datasets.Dataset.from_generator(gen)
