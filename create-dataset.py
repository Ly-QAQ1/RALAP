import os
from pathlib import Path

import pandas as pd
from transformers import set_seed

try:
    from data.personal_dataset import (
        PersonalDataset,
        convert_to_dataset,
        get_label_names,
        load_label_space,
    )
    from utils.tokenizer_utils import load_or_build_dep_processor
except ModuleNotFoundError:
    from promptlearn.DEPMultiRAG.data.personal_dataset import (
        PersonalDataset,
        convert_to_dataset,
        get_label_names,
        load_label_space,
    )
    from promptlearn.DEPMultiRAG.utils.tokenizer_utils import load_or_build_dep_processor

set_seed(30)

DATASET_NAME = os.getenv("RALAP_DATASET_NAME", "food101").strip().lower()
DATASET_ROOT = Path(os.getenv("RAGPT_DATA_ROOT", "dataset"))
MEMORY_BANK_ROOT = Path(
    os.getenv("RAGPT_MEMORY_BANK_ROOT", str(DATASET_ROOT / "memory_bank"))
)
OUTPUT_ROOT = Path(os.getenv("HF_DATA_OUTPUT_ROOT", "data"))
TOKENIZER_OUTPUT_DIR = Path(os.getenv("TOKENIZER_OUTPUT_DIR", "output/DEP-tokenizer"))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
LLM_MAX_LENGTH = int(os.getenv("LLM_MAX_LENGTH", "2048"))
TARGET_MAX_LENGTH = int(os.getenv("TARGET_MAX_LENGTH", "256"))
PROMPT_TOKEN_COUNT = int(os.getenv("PROMPT_TOKEN_COUNT", "3"))
EXPLANATION_MAX_CHARS = int(os.getenv("EXPLANATION_MAX_CHARS", "256"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.1"))
QUERY_MODES = tuple(
    mode.strip().lower()
    for mode in os.getenv("QUERY_MODES", "text,image").split(",")
    if mode.strip()
)
AUTO_PREPARE_RETRIEVAL = os.getenv("AUTO_PREPARE_RETRIEVAL", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_retrieval_preparation_attempted = False

REQUIRED_RETRIEVAL_COLUMNS = [
    "item_id",
    "text",
    "label",
    "t2t_id_list",
    "t2t_sims_list",
    "t2t_label_list",
    "i2i_id_list",
    "i2i_sims_list",
    "i2i_label_list",
]


def ensure_path_exists(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found: {path}\n"
            "Please prepare the RAGPT dataset and retrieval files first "
            "(for example via init_dataRAGPT.py)."
        )


def get_missing_retrieval_columns(df: pd.DataFrame):
    return [col for col in REQUIRED_RETRIEVAL_COLUMNS if col not in df.columns]


def prepare_retrieval_data_once():
    global _retrieval_preparation_attempted
    if _retrieval_preparation_attempted:
        return
    _retrieval_preparation_attempted = True

    if not AUTO_PREPARE_RETRIEVAL:
        return

    try:
        from init_dataRAGPT import load_cfg, main as init_data_main
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import init_dataRAGPT.py to prepare retrieval columns. "
            "Run `python init_dataRAGPT.py` manually, or set "
            "AUTO_PREPARE_RETRIEVAL=0 to keep the explicit failure."
        ) from exc

    cfg = load_cfg()
    cfg.data_para.dataset_name = DATASET_NAME
    print(
        f"Retrieval columns are missing for {DATASET_NAME}. "
        "Running init_dataRAGPT.py once to build memory banks and retrieval results..."
    )
    init_data_main(cfg)


def load_split(split: str) -> pd.DataFrame:
    path = DATASET_ROOT / DATASET_NAME / f"{split}.pkl"
    if not path.exists() and AUTO_PREPARE_RETRIEVAL:
        prepare_retrieval_data_once()
    ensure_path_exists(path, f"{split} split")
    df = pd.read_pickle(path)
    missing_columns = get_missing_retrieval_columns(df)
    if missing_columns and AUTO_PREPARE_RETRIEVAL:
        prepare_retrieval_data_once()
        df = pd.read_pickle(path)
        missing_columns = get_missing_retrieval_columns(df)

    if missing_columns:
        raise ValueError(
            f"{path} is missing retrieval columns: {missing_columns}. "
            "Please run `python init_dataRAGPT.py` before building the LLM dataset, "
            "or leave AUTO_PREPARE_RETRIEVAL=1 so run-create can prepare them automatically."
        )
    return df.reset_index(drop=True)


def make_validation_from_train(train_df: pd.DataFrame, val_ratio: float):
    shuffled = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_size = max(1, int(len(shuffled) * val_ratio))
    if val_size >= len(shuffled):
        val_size = max(1, len(shuffled) - 1)
    val_df = shuffled.iloc[:val_size].reset_index(drop=True)
    new_train_df = shuffled.iloc[val_size:].reset_index(drop=True)
    return new_train_df, val_df


def build_hf_dataset(split_name: str, dataframe: pd.DataFrame, tokenizer, training: bool):
    dataset = PersonalDataset(
        dataframe=dataframe,
        dataset_name=DATASET_NAME,
        llm_tokenizer=tokenizer,
        memory_root=str(MEMORY_BANK_ROOT),
        dataset_root=str(DATASET_ROOT),
        max_length=LLM_MAX_LENGTH,
        max_target_length=TARGET_MAX_LENGTH,
        prompt_token_count=PROMPT_TOKEN_COUNT,
        query_modes=QUERY_MODES,
        explanation_max_chars=EXPLANATION_MAX_CHARS,
        training=training,
    )
    hf_dataset = convert_to_dataset(dataset)
    output_path = OUTPUT_ROOT / f"dataset_{split_name}"
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
    hf_dataset.save_to_disk(str(output_path))
    print(f"Saved {split_name} dataset to {output_path} with {len(hf_dataset)} samples.")


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TOKENIZER_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    valid_path = DATASET_ROOT / DATASET_NAME / "valid.pkl"
    if valid_path.exists():
        valid_df = load_split("valid")
    else:
        train_df, valid_df = make_validation_from_train(train_df, VAL_RATIO)
        print(
            f"No explicit valid split found for {DATASET_NAME}. "
            f"Created a validation split from train: train={len(train_df)}, val={len(valid_df)}."
        )
    test_df = load_split("test")

    ensure_path_exists(MEMORY_BANK_ROOT / DATASET_NAME / "text", "Text memory bank")
    ensure_path_exists(MEMORY_BANK_ROOT / DATASET_NAME / "image", "Image memory bank")
    label_token_count = len(get_label_names(load_label_space(DATASET_NAME, DATASET_ROOT)))

    processor = load_or_build_dep_processor(
        base_model_name=LLM_MODEL_NAME,
        processor_path=str(TOKENIZER_OUTPUT_DIR),
        prompt_token_count=PROMPT_TOKEN_COUNT,
        label_token_count=label_token_count,
        save_if_rebuilt=True,
    )
    tokenizer = processor.tokenizer

    print(
        f"Building LLM datasets for {DATASET_NAME} with query modes: {', '.join(QUERY_MODES)}"
    )
    build_hf_dataset("train", train_df, tokenizer, training=True)
    build_hf_dataset("val", valid_df, tokenizer, training=False)
    build_hf_dataset("test", test_df, tokenizer, training=False)
