import os

from transformers import AutoProcessor, AutoTokenizer


DEFAULT_QWENVL_MIN_PIXELS = 56 * 56
DEFAULT_QWENVL_MAX_PIXELS = 256 * 28 * 28


def build_dep_special_tokens(prompt_token_count: int, label_token_count: int = 0):
    label_tokens = [f"[LAB_ATTN_{i}]" for i in range(label_token_count)]
    return [f"[HIS_TOKEN_{i}]" for i in range(prompt_token_count)] + [
        f"[DIFF_TOKEN_{i}]" for i in range(prompt_token_count)
    ] + label_tokens + [
        "<his_token_start>",
        "<his_token_end>",
        "<diff_token_start>",
        "<diff_token_end>",
        "<label_attn_start>",
        "<label_attn_end>",
    ]


def add_dep_special_tokens(tokenizer, prompt_token_count, label_token_count):
    special_tokens = build_dep_special_tokens(prompt_token_count, label_token_count)
    existing_tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    combined_tokens = list(dict.fromkeys(existing_tokens + special_tokens))
    vocab = tokenizer.get_vocab()
    missing_tokens = [token for token in combined_tokens if token not in vocab]
    if missing_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": combined_tokens})
    return tokenizer


def load_or_build_dep_tokenizer(
    base_model_name: str,
    tokenizer_path: str,
    prompt_token_count: int,
    label_token_count: int = 0,
    save_if_rebuilt: bool = True,
):
    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            tokenizer = None

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    add_dep_special_tokens(tokenizer, prompt_token_count, label_token_count)

    if save_if_rebuilt and tokenizer_path:
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)

    return tokenizer


def _prepare_dep_tokenizer(tokenizer, prompt_token_count, label_token_count):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    add_dep_special_tokens(tokenizer, prompt_token_count, label_token_count)
    return tokenizer


def _read_optional_int_env(name: str, default: int):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _apply_qwenvl_pixel_budget(processor, min_pixels=None, max_pixels=None):
    min_pixels = DEFAULT_QWENVL_MIN_PIXELS if min_pixels is None else int(min_pixels)
    max_pixels = DEFAULT_QWENVL_MAX_PIXELS if max_pixels is None else int(max_pixels)
    if min_pixels <= 0 or max_pixels <= 0:
        raise ValueError("Qwen2-VL pixel budget values must be positive integers.")
    if min_pixels > max_pixels:
        raise ValueError(
            "Qwen2-VL min pixel budget cannot exceed max pixel budget: "
            f"min={min_pixels}, max={max_pixels}."
        )

    for attr_name in ("image_processor", "video_processor"):
        visual_processor = getattr(processor, attr_name, None)
        if visual_processor is None or not hasattr(visual_processor, "size"):
            continue
        size = dict(getattr(visual_processor, "size") or {})
        size["shortest_edge"] = min_pixels
        size["longest_edge"] = max_pixels
        visual_processor.size = size
    return processor


def load_or_build_dep_processor(
    base_model_name: str,
    processor_path: str,
    prompt_token_count: int,
    label_token_count: int = 0,
    save_if_rebuilt: bool = True,
):
    processor = None
    if processor_path and os.path.exists(processor_path):
        try:
            processor = AutoProcessor.from_pretrained(processor_path)
        except Exception:
            processor = None

    if processor is None:
        processor = AutoProcessor.from_pretrained(base_model_name)

    if not hasattr(processor, "tokenizer"):
        processor = AutoProcessor.from_pretrained(base_model_name)
        if not hasattr(processor, "tokenizer"):
            raise ValueError(
                "Qwen2-VL processor must expose a tokenizer. "
                "Please point TOKENIZER_PATH to a Qwen2-VL processor directory."
            )

    _prepare_dep_tokenizer(
        processor.tokenizer,
        prompt_token_count,
        label_token_count,
    )
    _apply_qwenvl_pixel_budget(
        processor,
        min_pixels=_read_optional_int_env("QWENVL_MIN_PIXELS", DEFAULT_QWENVL_MIN_PIXELS),
        max_pixels=_read_optional_int_env("QWENVL_MAX_PIXELS", DEFAULT_QWENVL_MAX_PIXELS),
    )

    if save_if_rebuilt and processor_path:
        os.makedirs(processor_path, exist_ok=True)
        processor.save_pretrained(processor_path)

    return processor
