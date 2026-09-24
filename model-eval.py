import argparse
import glob
import json
import os
import re
import sys

def _early_get_cli_value(flag, default=None):
    if flag not in sys.argv:
        return default
    idx = sys.argv.index(flag)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]


_early_gpu = _early_get_cli_value("--gpu", os.environ.get("GPU"))
if _early_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_gpu

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_CACHE_DIR = os.path.join(_SCRIPT_DIR, ".cache")
os.environ.setdefault("HF_HOME", os.path.join(_LOCAL_CACHE_DIR, "huggingface"))
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(os.environ["HF_HOME"], "modules"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_LOCAL_CACHE_DIR, "huggingface", "datasets"))
os.environ.setdefault("HF_EVALUATE_CACHE", os.path.join(_LOCAL_CACHE_DIR, "huggingface", "evaluate"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.join(_LOCAL_CACHE_DIR, "huggingface", "metrics"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_LOCAL_CACHE_DIR, "matplotlib"))
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


def ensure_writable_cache_dir(env_name, default_dir):
    cache_dir = os.environ.get(env_name, default_dir)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        probe_path = os.path.join(cache_dir, ".write_test")
        with open(probe_path, "w", encoding="utf-8") as probe:
            probe.write("")
        os.remove(probe_path)
    except OSError:
        cache_dir = default_dir
        os.environ[env_name] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


for _cache_dir in (os.environ["HF_HOME"], os.environ["HF_MODULES_CACHE"]):
    os.makedirs(_cache_dir, exist_ok=True)
for _env_name, _default_dir in (
    ("HF_DATASETS_CACHE", os.path.join(_LOCAL_CACHE_DIR, "huggingface", "datasets")),
    ("HF_EVALUATE_CACHE", os.path.join(_LOCAL_CACHE_DIR, "huggingface", "evaluate")),
    ("HF_METRICS_CACHE", os.path.join(_LOCAL_CACHE_DIR, "huggingface", "metrics")),
    ("MPLCONFIGDIR", os.path.join(_LOCAL_CACHE_DIR, "matplotlib")),
):
    ensure_writable_cache_dir(_env_name, _default_dir)

import evaluate
import numpy as np
import torch
import torch.distributed as dist
import warnings
from bert_score.scorer import BERTScorer
from datasets import load_from_disk
from safetensors.torch import safe_open
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    BlipForConditionalGeneration,
    logging,
    set_seed,
)

try:
    from model.personal_model import DEPModel
    from data.personal_dataset import QWENVL_IMAGE_TOKEN, load_label_space
    from utils.utils import postprocess_output
    from utils.multimodal_collator import build_qwenvl_generation_batch
    from utils.tokenizer_utils import load_or_build_dep_processor, load_or_build_dep_tokenizer
except ModuleNotFoundError:
    from promptlearn.DEPMultiRAG.model.personal_model import DEPModel
    from promptlearn.DEPMultiRAG.data.personal_dataset import QWENVL_IMAGE_TOKEN, load_label_space
    from promptlearn.DEPMultiRAG.utils.utils import postprocess_output
    from promptlearn.DEPMultiRAG.utils.multimodal_collator import build_qwenvl_generation_batch
    from promptlearn.DEPMultiRAG.utils.tokenizer_utils import load_or_build_dep_processor, load_or_build_dep_tokenizer

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

set_seed(42)


def str_to_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


DSP_ABLATION_MODES = {
    "full": (True, True),
    "sae_only": (True, False),
    "label_only": (False, True),
    "no_dsp": (False, False),
}


def resolve_dsp_components(mode):
    normalized_mode = str(mode or "full").strip().lower()
    if normalized_mode not in DSP_ABLATION_MODES:
        raise ValueError(
            "DSP ablation mode must be one of: full, sae_only, label_only, no_dsp."
        )
    return DSP_ABLATION_MODES[normalized_mode]


def validate_ablation_modes(args):
    dsp_mode = getattr(args, "dsp_ablation_mode", "full")
    resolve_dsp_components(dsp_mode)
    if getattr(args, "ablation_mode", "none") != "none" and dsp_mode != "full":
        raise ValueError(
            "EVAL_ABLATION_MODE and DSP_ABLATION_MODE cannot be combined. "
            "Run modality/input ablations and DSP component ablations separately."
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["infer", "eval"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--model_path", default="output")
    parser.add_argument("--base_model_name", default=os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct"))
    parser.add_argument("--tokenizer_path", default="output/DEP-tokenizer")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--dataset_root", default=os.getenv("RAGPT_DATA_ROOT", "dataset"))
    parser.add_argument("--dataset_name", default=os.getenv("RALAP_DATASET_NAME", ""))
    parser.add_argument("--dataset_path", default=os.getenv("EVAL_DATA_DIR", ""))
    parser.add_argument("--backend", default=os.getenv("EVAL_BACKEND", "auto"), choices=["auto", "vllm", "transformers"])
    parser.add_argument("--predictions_path", default="")
    parser.add_argument("--constrained_predictions_path", default=os.getenv("CONSTRAINED_PREDICTIONS_PATH", ""))
    parser.add_argument(
        "--label_attn_report_path",
        default=os.getenv("LABEL_ATTN_REPORT_PATH", ""),
        help="Optional JSONL path for instance-level and label-level label attention scores.",
    )
    parser.add_argument(
        "--ablation_mode",
        default=os.getenv("EVAL_ABLATION_MODE", "none"),
        choices=["none", "text_only", "image_caption", "image_embedding", "text_embedding"],
        help="Run an ablation path instead of the full DEP retrieval soft-prompt inference.",
    )
    parser.add_argument(
        "--dsp_ablation_mode",
        default=os.getenv("DSP_ABLATION_MODE", "full"),
        choices=sorted(DSP_ABLATION_MODES),
        help="Inference-only component ablation for SAE and label-attention dynamic soft prompts.",
    )
    parser.add_argument(
        "--ablation_llm_path",
        default=os.getenv("ABLATION_LLM_PATH", ""),
        help="Plain text LLM path for ablations. Defaults to --model_path.",
    )
    parser.add_argument(
        "--caption_model_name",
        default=os.getenv("CAPTION_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct"),
        help="Image captioning model used by EVAL_ABLATION_MODE=image_caption.",
    )
    parser.add_argument(
        "--caption_cache_path",
        default=os.getenv("CAPTION_CACHE_PATH", ""),
        help="Optional JSON cache for image captions. Defaults to model_path/captions_<split>.json.",
    )
    parser.add_argument(
        "--caption_batch_size",
        type=int,
        default=int(os.getenv("CAPTION_BATCH_SIZE", "8")),
    )
    parser.add_argument(
        "--caption_max_new_tokens",
        type=int,
        default=int(os.getenv("CAPTION_MAX_NEW_TOKENS", "64")),
    )
    parser.add_argument(
        "--constrain_labels",
        action=argparse.BooleanOptionalAction,
        default=str_to_bool(os.getenv("CONSTRAIN_LABELS", "1")),
        help="Canonicalize generated Label lines to the dataset allowed label space.",
    )
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_input_length", type=int, default=int(os.getenv("EVAL_MAX_INPUT_LENGTH", "1024")))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("EVAL_BATCH_SIZE", "1")))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--limit", type=int, default=int(os.getenv("EVAL_LIMIT", "0")))
    parser.add_argument(
        "--dump_caption_cache_template",
        action="store_true",
        help="Write a sample_id -> empty caption JSON template for the selected dataset and exit.",
    )
    return parser.parse_args()


def get_dataset_path(args):
    if args.dataset_path:
        return args.dataset_path

    default_path = os.path.join(args.data_root, f"dataset_{args.split}")
    if os.path.isdir(default_path):
        return default_path

    if args.dataset_name:
        named_path = os.path.join(args.data_root, f"dataset_{args.split}_{args.dataset_name}")
        if os.path.isdir(named_path):
            return named_path
        raise FileNotFoundError(
            f"Directory {named_path} not found. "
            f"Use --dataset_path to specify the exact {args.split} dataset directory."
        )

    candidates = sorted(glob.glob(os.path.join(args.data_root, f"dataset_{args.split}_*")))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise FileNotFoundError(
            f"Directory {default_path} not found. Available {args.split} datasets: "
            f"{', '.join(candidates)}. Please pass --dataset_name or --dataset_path."
        )
    raise FileNotFoundError(f"Directory {default_path} not found")


def get_ablation_query_mode(ablation_mode):
    if ablation_mode in {"text_only", "text_embedding"}:
        return "text"
    if ablation_mode in {"image_caption", "image_embedding"}:
        return "image"
    return None


def filter_dataset_for_ablation(args, personal_dataset):
    query_mode = get_ablation_query_mode(args.ablation_mode)
    if query_mode is None:
        return personal_dataset
    if "query_mode" not in personal_dataset.column_names:
        raise ValueError(
            f"EVAL_ABLATION_MODE={args.ablation_mode} requires a dataset with a query_mode column."
        )

    indices = [
        idx
        for idx, mode in enumerate(personal_dataset["query_mode"])
        if mode == query_mode
    ]
    if not indices:
        raise ValueError(
            f"EVAL_ABLATION_MODE={args.ablation_mode} selected query_mode={query_mode}, "
            "but no matching samples were found in the evaluation dataset."
        )

    filtered_dataset = personal_dataset.select(indices)
    print(
        f"Ablation dataset filtered by query_mode={query_mode}: "
        f"{len(personal_dataset)} -> {len(filtered_dataset)}"
    )
    return filtered_dataset


def get_predictions_path(args):
    if args.predictions_path:
        return args.predictions_path
    ablation_suffix = "" if args.ablation_mode == "none" else f"_{args.ablation_mode}"
    dsp_mode = getattr(args, "dsp_ablation_mode", "full")
    dsp_suffix = "" if dsp_mode == "full" else f"_dsp_{dsp_mode}"
    return os.path.join(
        args.model_path,
        f"predictions_{args.split}{ablation_suffix}{dsp_suffix}.txt",
    )


def get_constrained_predictions_path(args):
    if args.constrained_predictions_path:
        return args.constrained_predictions_path
    predictions_path = get_predictions_path(args)
    root, ext = os.path.splitext(predictions_path)
    return f"{root}_constrained{ext or '.txt'}"


def write_predictions(path, predictions):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n---------------------------------\n")


def write_jsonl(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_backend(args):
    dsp_mode = getattr(args, "dsp_ablation_mode", "full")
    if dsp_mode != "full":
        if args.backend == "vllm":
            raise ValueError(
                "DSP component ablations require the Transformers backend; "
                "vLLM does not propagate the DEP component switches."
            )
        return "transformers"
    if args.backend != "auto":
        return args.backend
    config = AutoConfig.from_pretrained(args.model_path)
    architectures = getattr(config, "architectures", []) or []
    if "DEPModel" in architectures:
        return "transformers"
    if getattr(config, "model_type", "") in {"qwen2_vl", "qwen2_5_vl"}:
        return "transformers"
    return "vllm"


def dataset_has_label_attention(personal_dataset):
    required_columns = {
        "label_attn_target_emb",
        "label_attn_retrieved_emb",
        "label_attn_label_mask",
        "label_attn_valid_mask",
    }
    return required_columns.issubset(set(personal_dataset.column_names))


def dataset_has_image_queries(personal_dataset):
    return "query_mode" in personal_dataset.column_names and any(
        mode == "image" for mode in personal_dataset["query_mode"]
    )


def get_model_rag_config(model_path):
    config = AutoConfig.from_pretrained(model_path)
    rag_config = {
        "rag_embed_size": getattr(config, "rag_embed_size", None),
        "sparse_hidden_size": getattr(config, "sparse_hidden_size", None),
        "prompt_token_count": getattr(config, "prompt_token_count", None),
        "label_token_count": getattr(config, "label_token_count", None),
        "label_attn_size": getattr(config, "label_attn_size", None),
    }
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        with safe_open(safetensors_path, framework="pt", device="cpu") as state:
            keys = set(state.keys())
            if "sae.encoder.0.weight" in keys:
                encoder_shape = state.get_tensor("sae.encoder.0.weight").shape
                rag_config["rag_embed_size"] = int(encoder_shape[1])
                rag_config["sparse_hidden_size"] = int(encoder_shape[0])
            if "label_attention.label_embedding.weight" in keys:
                label_embedding_shape = state.get_tensor(
                    "label_attention.label_embedding.weight"
                ).shape
                rag_config["label_token_count"] = int(label_embedding_shape[0])
                rag_config["label_attn_size"] = int(label_embedding_shape[1])
    return rag_config


def strip_label_attention_table(prompt):
    prompt = re.sub(
        r"<label_attn_start>\n.*?\n<label_attn_end>\n?",
        "",
        prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        r"\[Label Attention Table\]\n\| Label \| Attention \|.*?(?=\n[A-Za-z\[]|$)",
        "",
        prompt,
        flags=re.DOTALL,
    )
    return prompt


def strip_sae_prompt_blocks(prompt):
    for pattern in (
        r"^\[Self Prompt\]:\s*<his_token_start>.*?<his_token_end>\s*\n?",
        r"^\[Difference Prompt\]:\s*<diff_token_start>.*?<diff_token_end>\s*\n?",
    ):
        prompt = re.sub(pattern, "", prompt, flags=re.MULTILINE)
    return prompt


def apply_dsp_prompt_ablation(prompt, mode):
    use_sae_dsp, use_label_dsp = resolve_dsp_components(mode)
    if not use_sae_dsp:
        prompt = strip_sae_prompt_blocks(prompt)
    if not use_label_dsp:
        prompt = strip_label_attention_table(prompt)
    return prompt


def add_dsp_generation_inputs(
    generation_kwargs,
    mode,
    his_diff_emb=None,
    label_attention_inputs=None,
):
    use_sae_dsp, use_label_dsp = resolve_dsp_components(mode)
    generation_kwargs = dict(generation_kwargs)
    generation_kwargs["use_sae_dsp"] = use_sae_dsp
    generation_kwargs["use_label_dsp"] = use_label_dsp
    if use_sae_dsp:
        if his_diff_emb is None:
            raise ValueError("SAE DSP is enabled but his_diff_emb was not provided.")
        generation_kwargs["his_diff_emb"] = his_diff_emb
    if use_label_dsp and label_attention_inputs:
        generation_kwargs.update(label_attention_inputs)
    return generation_kwargs


def get_dataset_rag_shape(personal_dataset):
    if len(personal_dataset) == 0:
        return None, None
    sample = personal_dataset[0]["his_diff_emb"]
    return len(sample), len(sample[0]) if sample else None


def validate_dep_checkpoint(args, personal_dataset):
    vector_count, embed_size = get_dataset_rag_shape(personal_dataset)
    model_rag_config = get_model_rag_config(args.model_path)
    configured_embed_size = model_rag_config["rag_embed_size"] or int(os.getenv("RAG_EMBED_SIZE", "768"))
    configured_prompt_count = model_rag_config["prompt_token_count"] or int(os.getenv("PROMPT_TOKEN_COUNT", "3"))

    if embed_size is not None and configured_embed_size != embed_size:
        raise ValueError(
            "DEP checkpoint and evaluation dataset have incompatible retrieval embedding sizes: "
            f"checkpoint expects {configured_embed_size}, dataset provides {embed_size}. "
            "Please switch to a checkpoint trained with the same RAG_EMBED_SIZE, "
            "or rebuild the dataset/memory bank to match the checkpoint."
        )
    if vector_count is not None and vector_count not in {configured_prompt_count + 1, configured_prompt_count * 2}:
        raise ValueError(
            "DEP checkpoint and evaluation dataset have incompatible prompt vector counts: "
            f"checkpoint prompt_token_count={configured_prompt_count}, dataset his_diff_emb has "
            f"{vector_count} vectors."
        )

    configured_label_count = model_rag_config["label_token_count"] or 0
    if dataset_has_label_attention(personal_dataset):
        dataset_label_count = len(personal_dataset[0]["label_attn_label_mask"][0])
        if configured_label_count not in {0, dataset_label_count}:
            raise ValueError(
                "DEP checkpoint and evaluation dataset have incompatible label attention sizes: "
                f"checkpoint label_token_count={configured_label_count}, dataset has "
                f"{dataset_label_count} labels."
            )
        configured_label_count = dataset_label_count

    return (
        configured_embed_size,
        model_rag_config["sparse_hidden_size"] or int(os.getenv("SPARSE_HIDDEN_SIZE", "512")),
        configured_prompt_count,
        configured_label_count,
        model_rag_config["label_attn_size"] or int(os.getenv("LABEL_ATTN_SIZE", "256")),
    )


def generate_with_vllm(args, personal_dataset):
    if dataset_has_image_queries(personal_dataset):
        raise ValueError(
            "The vLLM backend does not support DEP Qwen2-VL raw image visual tokens. "
            "Use --backend transformers for multimodal inference."
        )
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        temperature=args.temperature,
        top_p=0.95,
    )
    llm = LLM(
        args.model_path,
        tokenizer=args.tokenizer_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )

    prompts = personal_dataset["inp_str"]
    his_diff_embs = [torch.tensor(item) for item in personal_dataset["his_diff_emb"]]
    outputs = llm.generate(
        prompts,
        his_diff_embs=his_diff_embs,
        sampling_params=sampling_params,
    )
    return [pred.outputs[0].text.strip() for pred in outputs]


def _prepare_generation_features(args, personal_dataset, start_idx, end_idx, prompts):
    batch_features = []
    for sample_idx in range(start_idx, end_idx):
        feature = dict(personal_dataset[sample_idx])
        feature["inp_str"] = prompts[sample_idx]
        if (
            (feature.get("query_mode") == "image" or QWENVL_IMAGE_TOKEN in feature["inp_str"])
            and not feature.get("image_path")
        ):
            feature["image_path"] = find_sample_image(args, feature["sample_id"])
        batch_features.append(feature)
    return batch_features


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def generate_with_transformers(args, personal_dataset, processor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = processor.tokenizer
    (
        rag_embed_size,
        sparse_hidden_size,
        prompt_token_count,
        label_token_count,
        label_attn_size,
    ) = validate_dep_checkpoint(args, personal_dataset)
    dsp_mode = getattr(args, "dsp_ablation_mode", "full")
    use_sae_dsp, use_label_dsp = resolve_dsp_components(dsp_mode)
    model_rag_config = get_model_rag_config(args.model_path)
    checkpoint_has_label_attention = bool(model_rag_config["label_token_count"])
    dataset_has_label_dsp = dataset_has_label_attention(personal_dataset)
    if dsp_mode == "label_only" and not checkpoint_has_label_attention:
        raise ValueError(
            "DSP_ABLATION_MODE=label_only requires a checkpoint with label-attention weights."
        )
    if dsp_mode == "label_only" and not dataset_has_label_dsp:
        raise ValueError(
            "DSP_ABLATION_MODE=label_only requires label-attention columns in the evaluation dataset."
        )
    if not checkpoint_has_label_attention:
        label_token_count = 0
    model = DEPModel.from_pretrained(
        args.model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        rag_embed_size=rag_embed_size,
        sparse_hidden_size=sparse_hidden_size,
        prompt_token_count=prompt_token_count,
        label_token_count=label_token_count,
        label_attn_size=label_attn_size,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.to(device)
    model.eval()

    predictions = []
    prompts = [
        apply_dsp_prompt_ablation(prompt, dsp_mode)
        for prompt in personal_dataset["inp_str"]
    ]
    if not checkpoint_has_label_attention:
        prompts = [strip_label_attention_table(prompt) for prompt in prompts]
    his_diff_embs = personal_dataset["his_diff_emb"] if use_sae_dsp else None
    has_label_attention = (
        use_label_dsp and checkpoint_has_label_attention and dataset_has_label_dsp
    )
    label_attn_report_rows = []
    allowed_labels = (
        get_allowed_labels(get_effective_dataset_name(args), args.dataset_root)
        if args.label_attn_report_path and has_label_attention
        else []
    )
    do_sample = args.temperature > 0

    for start_idx in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        end_idx = min(start_idx + args.batch_size, len(prompts))
        batch_features = _prepare_generation_features(
            args,
            personal_dataset,
            start_idx,
            end_idx,
            prompts,
        )
        batch_his_diff = None
        if use_sae_dsp:
            batch_his_diff = torch.tensor(
                his_diff_embs[start_idx:end_idx],
                dtype=torch.bfloat16,
                device=device,
            )
        encoded = build_qwenvl_generation_batch(
            processor,
            batch_features,
            args.max_input_length,
        )
        encoded = move_batch_to_device(encoded, device)

        generation_kwargs = {
            **encoded,
            "max_new_tokens": args.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        label_attention_inputs = None
        if has_label_attention:
            batch_label_attn_target = torch.tensor(
                personal_dataset["label_attn_target_emb"][start_idx:end_idx],
                dtype=torch.bfloat16,
                device=device,
            )
            batch_label_attn_retrieved = torch.tensor(
                personal_dataset["label_attn_retrieved_emb"][start_idx:end_idx],
                dtype=torch.bfloat16,
                device=device,
            )
            batch_label_attn_mask = torch.tensor(
                personal_dataset["label_attn_label_mask"][start_idx:end_idx],
                dtype=torch.bfloat16,
                device=device,
            )
            batch_label_attn_valid = torch.tensor(
                personal_dataset["label_attn_valid_mask"][start_idx:end_idx],
                dtype=torch.bfloat16,
                device=device,
            )
            label_attention_inputs = {
                "label_attn_target_emb": batch_label_attn_target,
                "label_attn_retrieved_emb": batch_label_attn_retrieved,
                "label_attn_label_mask": batch_label_attn_mask,
                "label_attn_valid_mask": batch_label_attn_valid,
            }
            if "target_label_mask" in personal_dataset.column_names:
                label_attention_inputs["target_label_mask"] = torch.tensor(
                    personal_dataset["target_label_mask"][start_idx:end_idx],
                    dtype=torch.bfloat16,
                    device=device,
                )
            if args.label_attn_report_path and getattr(model, "label_attention", None) is not None:
                with torch.inference_mode():
                    _, label_scores, instance_scores, _ = model.label_attention(
                        batch_label_attn_target,
                        batch_label_attn_retrieved,
                        batch_label_attn_mask,
                        batch_label_attn_valid,
                    )
                batch_masks = batch_label_attn_mask.float().cpu().numpy()
                batch_valid = batch_label_attn_valid.float().cpu().numpy()
                batch_instance_scores = instance_scores.float().cpu().numpy()
                batch_label_scores = label_scores.float().cpu().numpy()
                for local_idx, sample_idx in enumerate(range(start_idx, end_idx)):
                    instances = []
                    for inst_idx, valid in enumerate(batch_valid[local_idx].tolist()):
                        inst_labels = [
                            allowed_labels[label_idx]
                            for label_idx, value in enumerate(batch_masks[local_idx, inst_idx].tolist())
                            if value > 0.5 and label_idx < len(allowed_labels)
                        ]
                        instances.append(
                            {
                                "instance_index": inst_idx,
                                "valid": bool(valid > 0.5),
                                "score": float(batch_instance_scores[local_idx, inst_idx]),
                                "labels": inst_labels,
                            }
                        )
                    label_scores_row = [
                        {
                            "label": allowed_labels[label_idx],
                            "score": float(score),
                        }
                        for label_idx, score in enumerate(batch_label_scores[local_idx].tolist())
                        if label_idx < len(allowed_labels)
                    ]
                    label_attn_report_rows.append(
                        {
                            "sample_id": str(personal_dataset["sample_id"][sample_idx]),
                            "query_mode": personal_dataset["query_mode"][sample_idx]
                            if "query_mode" in personal_dataset.column_names
                            else "",
                            "instances": instances,
                            "labels": label_scores_row,
                        }
                    )
        generation_kwargs = add_dsp_generation_inputs(
            generation_kwargs,
            dsp_mode,
            batch_his_diff,
            label_attention_inputs,
        )
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = 0.95

        with torch.inference_mode():
            generated_ids = model.generate(**generation_kwargs)

        prompt_length = encoded["input_ids"].shape[1]
        generated_texts = tokenizer.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
        )
        predictions.extend(text.strip() for text in generated_texts)

    if args.label_attn_report_path and label_attn_report_rows:
        write_jsonl(args.label_attn_report_path, label_attn_report_rows)
        print(f"Saved label attention report to {args.label_attn_report_path}")
    return predictions


def get_special_prompt_tokens(prompt_token_count):
    self_prompt_tokens = " ".join(f"[HIS_TOKEN_{idx}]" for idx in range(prompt_token_count))
    diff_prompt_tokens = " ".join(f"[DIFF_TOKEN_{idx}]" for idx in range(prompt_token_count))
    return self_prompt_tokens, diff_prompt_tokens


def get_embedding_ablation_modality(args):
    if args.ablation_mode == "text_embedding":
        return "text"
    if args.ablation_mode == "image_embedding":
        return "image"
    raise ValueError(f"Unsupported embedding ablation mode: {args.ablation_mode}")


def build_embedding_ablation_prompts(args, personal_dataset, tokenizer, prompt_token_count):
    modality = get_embedding_ablation_modality(args)
    modality_title = "Text" if modality == "text" else "Image"
    system_prompt = build_ablation_system_prompt(args)
    self_prompt_tokens, diff_prompt_tokens = get_special_prompt_tokens(prompt_token_count)
    prompts = []
    for _ in personal_dataset["sample_id"]:
        user_prompt = (
            f"[Dataset]: {get_effective_dataset_name(args)}\n"
            f"[Available Modality]: {modality} embedding only\n"
            f"[Active Retrieval Query]: {modality}\n"
            f"[Target {modality_title} Embedding]: The target sample evidence is provided through the dynamic soft prompt.\n"
            f"[Self Prompt]: <his_token_start>{self_prompt_tokens}<his_token_end>\n"
            f"[Difference Prompt]: <diff_token_start>{diff_prompt_tokens}<diff_token_end>\n"
            "Predict the label using only the allowed label names and provide a concise explanation. "
            "Do not output any label outside the allowed label list."
        )
        prompts.append(apply_chat_template(tokenizer, system_prompt, user_prompt))
    return prompts


def build_target_embedding_inputs(personal_dataset, prompt_token_count):
    his_diff_embs = []
    for context_emb in personal_dataset["his_diff_emb"]:
        target_emb = np.asarray(context_emb[0], dtype=np.float32)
        zero_emb = np.zeros_like(target_emb, dtype=np.float32)
        sae_input = [target_emb.copy() for _ in range(prompt_token_count)] + [
            zero_emb.copy() for _ in range(prompt_token_count)
        ]
        his_diff_embs.append(sae_input)
    return his_diff_embs


def generate_with_embedding_ablation(args, personal_dataset, tokenizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (
        rag_embed_size,
        sparse_hidden_size,
        prompt_token_count,
        _,
        _,
    ) = validate_dep_checkpoint(args, personal_dataset)
    model = DEPModel.from_pretrained(
        args.model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        rag_embed_size=rag_embed_size,
        sparse_hidden_size=sparse_hidden_size,
        prompt_token_count=prompt_token_count,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.to(device)
    model.eval()

    prompts = build_embedding_ablation_prompts(args, personal_dataset, tokenizer, prompt_token_count)
    his_diff_embs = build_target_embedding_inputs(personal_dataset, prompt_token_count)
    predictions = []
    do_sample = args.temperature > 0
    for start_idx in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        end_idx = min(start_idx + args.batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_his_diff = torch.tensor(
            his_diff_embs[start_idx:end_idx],
            dtype=torch.bfloat16,
            device=device,
        )
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs = {
            **encoded,
            "his_diff_emb": batch_his_diff,
            "max_new_tokens": args.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = 0.95
        with torch.inference_mode():
            generated_ids = model.generate(**generation_kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        generated_texts = tokenizer.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
        )
        predictions.extend(text.strip() for text in generated_texts)
    return predictions


def get_dataset_label_token_count(personal_dataset):
    if len(personal_dataset) > 0 and "target_label_mask" in personal_dataset.column_names:
        return len(personal_dataset[0]["target_label_mask"])
    return 0


def get_effective_dataset_name(args):
    if args.dataset_name:
        return args.dataset_name.lower()
    dataset_path = (args.dataset_path or "").lower()
    for dataset_name in ("mmimdb", "hatememes", "food101"):
        if dataset_name in dataset_path:
            return dataset_name
    return ""


def resolve_local_hf_model_path(model_name_or_path):
    expanded_path = os.path.expanduser(model_name_or_path)
    if os.path.exists(expanded_path):
        return expanded_path
    if "/" not in model_name_or_path:
        return model_name_or_path

    cache_dir_name = "models--" + model_name_or_path.replace("/", "--")
    candidate_hub_roots = [
        os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
        os.path.join(os.environ.get("HF_HOME", ""), "hub"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/data01/lyj/.cache/huggingface/hub",
    ]
    for hub_root in candidate_hub_roots:
        if not hub_root:
            continue
        model_cache_dir = os.path.join(hub_root, cache_dir_name)
        snapshots_dir = os.path.join(model_cache_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        snapshots = [
            os.path.join(snapshots_dir, snapshot)
            for snapshot in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, snapshot))
        ]
        if snapshots:
            return max(snapshots, key=os.path.getmtime)
    return model_name_or_path


def get_ablation_llm_path(args):
    return resolve_local_hf_model_path(args.ablation_llm_path or args.model_path)


def load_plain_llm_tokenizer(args, llm_path):
    tokenizer_source = args.tokenizer_path if args.tokenizer_path and os.path.exists(args.tokenizer_path) else llm_path
    tokenizer = load_or_build_dep_tokenizer(
        base_model_name=llm_path,
        tokenizer_path=tokenizer_source,
        prompt_token_count=int(os.getenv("PROMPT_TOKEN_COUNT", "3")),
        save_if_rebuilt=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_ablation_system_prompt(args):
    dataset_name = get_effective_dataset_name(args)
    allowed_labels = get_allowed_labels(dataset_name, args.dataset_root)
    label_line = ", ".join(allowed_labels) if allowed_labels else "the dataset allowed labels"
    return (
        f"You are a multimodal classification assistant for the {dataset_name or 'target'} dataset.\n"
        f"Allowed labels: {label_line}\n"
        "The Label field must contain only labels from the Allowed labels list. "
        "Do not invent, paraphrase, or output labels outside the allowed set.\n"
        "Always answer using exactly the following format:\n"
        "Label: <comma-separated labels from Allowed labels only>\n"
        "Explanation: <concise explanation>"
    )


def apply_chat_template(tokenizer, system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{system_prompt}\n\n{user_prompt}\n\nLabel:"


def parse_target_text_from_prompt(prompt):
    match = re.search(r"\[Target Text\]:\s*(.*?)(?:\n\[|$)", prompt, flags=re.DOTALL)
    if not match:
        return ""
    return " ".join(match.group(1).split())


def load_mmimdb_text(dataset_root, sample_id):
    metadata_path = os.path.join(dataset_root, "mmimdb", "meta_data", f"{sample_id}.json")
    if not os.path.exists(metadata_path):
        return ""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    plot_outline = metadata.get("plot outline")
    if plot_outline:
        return " ".join(str(plot_outline).split())
    plots = metadata.get("plot") or []
    if plots:
        return " ".join(str(plots[0]).split())
    return ""


def load_source_text(args, sample_id, fallback_prompt=""):
    dataset_name = get_effective_dataset_name(args)
    if dataset_name == "mmimdb":
        text = load_mmimdb_text(args.dataset_root, sample_id)
        if text:
            return text
    return parse_target_text_from_prompt(fallback_prompt)


def find_sample_image(args, sample_id):
    dataset_name = get_effective_dataset_name(args)
    image_dir = os.path.join(args.dataset_root, dataset_name, "image")
    for extension in ("jpeg", "jpg", "png", "webp"):
        image_path = os.path.join(image_dir, f"{sample_id}.{extension}")
        if os.path.exists(image_path):
            return image_path
    raise FileNotFoundError(f"Image file not found for sample_id={sample_id} under {image_dir}")


def get_caption_cache_path(args):
    if args.caption_cache_path:
        return args.caption_cache_path
    return os.path.join(args.model_path, f"captions_{args.split}.json")


def load_caption_cache(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_caption_cache(path, captions):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2, sort_keys=True)


def dump_caption_cache_template(args, personal_dataset):
    caption_cache_path = get_caption_cache_path(args)
    sample_ids = [str(sample_id) for sample_id in personal_dataset["sample_id"]]
    template = {sample_id: "" for sample_id in dict.fromkeys(sample_ids)}
    save_caption_cache(caption_cache_path, template)
    print(f"Saved caption cache template to {caption_cache_path}")


def load_caption_model(args, device):
    caption_model_path = resolve_local_hf_model_path(args.caption_model_name)
    try:
        config = AutoConfig.from_pretrained(caption_model_path)
        processor = AutoProcessor.from_pretrained(caption_model_path)
        model_type = getattr(config, "model_type", "")
        if model_type in {"qwen2_vl", "qwen2_5_vl"}:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            model = AutoModelForImageTextToText.from_pretrained(
                caption_model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            backend = "qwen_vl"
        else:
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            model = BlipForConditionalGeneration.from_pretrained(
                caption_model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            backend = "blip"
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the image captioning model. "
            "For EVAL_ABLATION_MODE=image_caption, set CAPTION_MODEL_NAME to a locally available "
            "Qwen-VL or BLIP-compatible caption model, or prefill CAPTION_CACHE_PATH with captions."
        ) from exc
    model.to(device)
    model.eval()
    return processor, model, backend


def move_caption_inputs_to_device(inputs, device):
    moved_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved_inputs[key] = value.to(device)
        else:
            moved_inputs[key] = value
    return moved_inputs


def generate_qwen_vl_captions(processor, caption_model, images, device, max_new_tokens):
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in one concise English sentence for movie genre "
                            "classification. Do not output genre labels."
                        ),
                    },
                ],
            }
        ]
        for image in images
    ]
    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_caption_inputs_to_device(inputs, device)
    with torch.inference_mode():
        generated_ids = caption_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=True,
    )


def generate_blip_captions(processor, caption_model, images, device, max_new_tokens):
    inputs = processor(images=images, return_tensors="pt")
    inputs = move_caption_inputs_to_device(inputs, device)
    with torch.inference_mode():
        generated_ids = caption_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def generate_missing_captions(args, sample_ids):
    from PIL import Image

    caption_cache_path = get_caption_cache_path(args)
    captions = load_caption_cache(caption_cache_path)
    missing_sample_ids = [
        sample_id
        for sample_id in dict.fromkeys(sample_ids)
        if sample_id not in captions or not str(captions[sample_id]).strip()
    ]
    if not missing_sample_ids:
        return captions
    if args.caption_cache_path:
        preview = ", ".join(missing_sample_ids[:10])
        suffix = " ..." if len(missing_sample_ids) > 10 else ""
        raise FileNotFoundError(
            f"Caption cache {caption_cache_path} is missing {len(missing_sample_ids)} sample ids: "
            f"{preview}{suffix}. "
            "Because --caption_cache_path was provided, image_caption mode will not fall back to loading "
            "a caption model. Please prefill the cache file with all required sample_id -> caption pairs."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor, caption_model, caption_backend = load_caption_model(args, device)
    for start_idx in tqdm(range(0, len(missing_sample_ids), args.caption_batch_size), desc="Captioning images"):
        batch_sample_ids = missing_sample_ids[start_idx : start_idx + args.caption_batch_size]
        images = []
        for sample_id in batch_sample_ids:
            image_path = find_sample_image(args, sample_id)
            images.append(Image.open(image_path).convert("RGB"))
        if caption_backend == "qwen_vl":
            generated_captions = generate_qwen_vl_captions(
                processor,
                caption_model,
                images,
                device,
                args.caption_max_new_tokens,
            )
        else:
            generated_captions = generate_blip_captions(
                processor,
                caption_model,
                images,
                device,
                args.caption_max_new_tokens,
            )
        for sample_id, caption in zip(batch_sample_ids, generated_captions):
            captions[sample_id] = " ".join(caption.strip().split())
        save_caption_cache(caption_cache_path, captions)
    return captions


def build_ablation_prompts(args, personal_dataset, tokenizer):
    system_prompt = build_ablation_system_prompt(args)
    sample_ids = [str(sample_id) for sample_id in personal_dataset["sample_id"]]
    prompts = []

    if args.ablation_mode == "text_only":
        for sample_id, fallback_prompt in zip(sample_ids, personal_dataset["inp_str"]):
            source_text = load_source_text(args, sample_id, fallback_prompt)
            user_prompt = (
                f"[Dataset]: {get_effective_dataset_name(args)}\n"
                "[Available Modality]: text only\n"
                f"[Target Text]: {source_text}\n"
                "Predict the label using only the allowed label names and provide a concise explanation. "
                "Do not output any label outside the allowed label list."
            )
            prompts.append(apply_chat_template(tokenizer, system_prompt, user_prompt))
        return prompts

    if args.ablation_mode == "image_caption":
        captions = generate_missing_captions(args, sample_ids)
        for sample_id in sample_ids:
            caption = captions.get(sample_id, "")
            user_prompt = (
                f"[Dataset]: {get_effective_dataset_name(args)}\n"
                "[Available Modality]: image caption only\n"
                f"[Image Caption]: {caption}\n"
                "Predict the label using only the allowed label names and provide a concise explanation. "
                "Do not output any label outside the allowed label list."
            )
            prompts.append(apply_chat_template(tokenizer, system_prompt, user_prompt))
        return prompts

    raise ValueError(f"Unsupported ablation mode: {args.ablation_mode}")


def generate_with_plain_llm(args, prompts, tokenizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    llm_path = get_ablation_llm_path(args)
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    predictions = []
    do_sample = args.temperature > 0
    for start_idx in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[start_idx : start_idx + args.batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs = {
            **encoded,
            "max_new_tokens": args.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = 0.95
        with torch.inference_mode():
            generated_ids = model.generate(**generation_kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        generated_texts = tokenizer.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
        )
        predictions.extend(text.strip() for text in generated_texts)
    return predictions


def generate_ablation_predictions(args, personal_dataset):
    if args.ablation_mode in {"image_embedding", "text_embedding"}:
        tokenizer = load_or_build_dep_tokenizer(
            base_model_name=args.base_model_name,
            tokenizer_path=args.tokenizer_path,
            prompt_token_count=int(os.getenv("PROMPT_TOKEN_COUNT", "3")),
            label_token_count=get_dataset_label_token_count(personal_dataset),
            save_if_rebuilt=False,
        )
        return generate_with_embedding_ablation(args, personal_dataset, tokenizer)

    llm_path = get_ablation_llm_path(args)
    tokenizer = load_plain_llm_tokenizer(args, llm_path)
    prompts = build_ablation_prompts(args, personal_dataset, tokenizer)
    return generate_with_plain_llm(args, prompts, tokenizer)


def extract_label(text):
    for line in text.splitlines():
        if line.lower().startswith("label:"):
            return line.split(":", 1)[1].strip()
    return ""


MMIMDB_LABEL_ALIASES = {
    "science fiction": ["Sci-Fi"],
    "science-fiction": ["Sci-Fi"],
    "sci fi": ["Sci-Fi"],
    "scifi": ["Sci-Fi"],
    "sports": ["Sport"],
    "historical": ["History"],
    "historical drama": ["History", "Drama"],
    "period": ["History"],
    "period drama": ["History", "Drama"],
    "suspense": ["Thriller"],
    "thrills": ["Thriller"],
    "biographical": ["Biography"],
    "biograpgy": ["Biography"],
    "bio": ["Biography"],
    "documentay": ["Documentary"],
    "documentary film": ["Documentary"],
    "nature documentary": ["Documentary"],
    "children": ["Family"],
    "children's": ["Family"],
    "romantic comedy": ["Romance", "Comedy"],
    "legal drama": ["Drama", "Crime"],
    "crime drama": ["Crime", "Drama"],
    "gangster": ["Crime"],
    "martial arts": ["Action"],
    "military": ["War"],
    "war film": ["War"],
    "time travel": ["Sci-Fi"],
}


def normalize_label_key(text):
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def get_allowed_labels(dataset_name, dataset_root):
    if not dataset_name:
        return []
    label_space = load_label_space(dataset_name.lower(), dataset_root)
    if isinstance(label_space, dict):
        return [str(label) for label in label_space.values()]
    return [str(label) for label in label_space]


def build_alias_map(dataset_name, allowed_labels):
    alias_map = {
        normalize_label_key(label): [label]
        for label in allowed_labels
    }
    if dataset_name.lower() == "mmimdb":
        for alias, labels in MMIMDB_LABEL_ALIASES.items():
            alias_map[normalize_label_key(alias)] = labels
    return alias_map


def split_generated_label_text(label_text):
    text = str(label_text)
    text = re.sub(r"\s+(and|&)\s+", ",", text, flags=re.IGNORECASE)
    for sep in (";", "/", "|"):
        text = text.replace(sep, ",")
    return [
        part.strip(" \t\r\n.[](){}'\"")
        for part in text.split(",")
        if part.strip(" \t\r\n.[](){}'\"")
    ]


def add_labels_in_order(target, seen, labels, allowed_set):
    for label in labels:
        if label not in allowed_set or label in seen:
            continue
        seen.add(label)
        target.append(label)


def canonicalize_label_text(label_text, allowed_labels, alias_map):
    allowed_set = set(allowed_labels)
    canonical_labels = []
    seen = set()
    alias_items = sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True)

    for part in split_generated_label_text(label_text):
        key = normalize_label_key(part)
        if not key:
            continue

        if key in alias_map:
            add_labels_in_order(canonical_labels, seen, alias_map[key], allowed_set)
            continue

        matched = False
        for alias_key, labels in alias_items:
            if not alias_key:
                continue
            if re.search(rf"\b{re.escape(alias_key)}\b", key):
                add_labels_in_order(canonical_labels, seen, labels, allowed_set)
                matched = True
        if matched:
            continue

    return canonical_labels


def rewrite_prediction_label(prediction, canonical_labels):
    label_text = ", ".join(canonical_labels) if canonical_labels else "None"
    lines = prediction.splitlines()
    for idx, line in enumerate(lines):
        if line.lower().startswith("label:"):
            lines[idx] = f"Label: {label_text}"
            return "\n".join(lines)
    return f"Label: {label_text}\n{prediction}".strip()


def constrain_prediction_labels(predictions, dataset_name, dataset_root):
    allowed_labels = get_allowed_labels(dataset_name, dataset_root)
    if not allowed_labels:
        return predictions
    alias_map = build_alias_map(dataset_name, allowed_labels)
    constrained_predictions = []
    for prediction in predictions:
        canonical_labels = canonicalize_label_text(
            extract_label(prediction),
            allowed_labels,
            alias_map,
        )
        constrained_predictions.append(
            rewrite_prediction_label(prediction, canonical_labels)
        )
    return constrained_predictions


def prefix_metrics(metrics, prefix):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def to_float(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_label_text(label_text):
    items = [part.strip().lower() for part in label_text.split(",") if part.strip()]
    if not items:
        return ""
    return ", ".join(sorted(items))


def parse_label_set(text):
    label_text = extract_label(text)
    return {part.strip().lower() for part in label_text.split(",") if part.strip()}


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def resolve_primary_metric(dataset_name):
    dataset_metric = {
        "hatememes": "auroc",
        "food101": "acc",
        "mmimdb": "multilabel",
    }.get(str(dataset_name).strip().lower())
    if dataset_metric is not None:
        return dataset_metric

    configured_metric = os.getenv("EVAL_PRIMARY_METRIC", "auto").strip().lower()
    if configured_metric not in {"", "auto"}:
        if configured_metric not in {"auroc", "acc", "multilabel"}:
            raise ValueError(
                "EVAL_PRIMARY_METRIC must be one of: auto, auroc, acc, multilabel."
            )
        return configured_metric

    return "multilabel"


def compute_binary_auroc(targets, scores):
    positive_count = sum(int(target == 1) for target in targets)
    negative_count = sum(int(target == 0) for target in targets)
    if positive_count == 0 or negative_count == 0:
        return 0.0

    ranked = sorted(zip(scores, targets), key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(
            int(target == 1) for _, target in ranked[index:end]
        )
        index = end

    return safe_divide(
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0,
        positive_count * negative_count,
    )


def compute_hatememes_auroc(predictions, references):
    targets = []
    scores = []
    for prediction, reference in zip(predictions, references):
        reference_labels = parse_label_set(reference)
        if reference_labels == {"hateful"}:
            targets.append(1)
        elif reference_labels == {"non-hateful"}:
            targets.append(0)
        else:
            raise ValueError(
                "Hateful Memes references must contain exactly one label: "
                "hateful or non-hateful."
            )

        prediction_labels = parse_label_set(prediction)
        if prediction_labels == {"hateful"}:
            scores.append(1.0)
        elif prediction_labels == {"non-hateful"}:
            scores.append(0.0)
        else:
            scores.append(0.5)

    return compute_binary_auroc(targets, scores)


def compute_single_label_accuracy(predictions, references):
    correct = 0
    for prediction, reference in zip(predictions, references):
        reference_labels = parse_label_set(reference)
        if len(reference_labels) != 1:
            raise ValueError("Single-label references must contain exactly one label.")
        prediction_labels = parse_label_set(prediction)
        correct += int(
            len(prediction_labels) == 1 and prediction_labels == reference_labels
        )
    return safe_divide(correct, len(references))


def compute_task_classification_metrics(dataset_name, predictions, references):
    dataset_name = str(dataset_name).strip().lower()
    if len(predictions) != len(references):
        raise ValueError("Prediction and reference counts must match.")
    if dataset_name == "hatememes":
        return {"auroc": compute_hatememes_auroc(predictions, references)}
    if dataset_name == "food101":
        return {"acc": compute_single_label_accuracy(predictions, references)}
    return compute_multilabel_metrics(predictions, references)


def compute_task_metrics_by_query_mode(
    dataset_name,
    query_modes,
    predictions,
    references,
):
    metrics_by_mode = {}
    for query_mode in ("image", "text"):
        indices = [idx for idx, mode in enumerate(query_modes) if mode == query_mode]
        if not indices:
            continue
        metrics_by_mode[query_mode] = {
            "sample_count": len(indices),
            **compute_task_classification_metrics(
                dataset_name,
                select_by_indices(predictions, indices),
                select_by_indices(references, indices),
            ),
        }
    return metrics_by_mode


def compute_multilabel_metrics(predictions, references):
    pred_sets = [parse_label_set(pred) for pred in predictions]
    ref_sets = [parse_label_set(ref) for ref in references]
    all_labels = sorted(set().union(*pred_sets, *ref_sets)) if pred_sets or ref_sets else []

    exact_matches = [int(pred == ref) for pred, ref in zip(pred_sets, ref_sets)]
    sample_jaccards = [
        safe_divide(len(pred & ref), len(pred | ref))
        for pred, ref in zip(pred_sets, ref_sets)
    ]
    sample_precisions = [
        safe_divide(len(pred & ref), len(pred))
        for pred, ref in zip(pred_sets, ref_sets)
    ]
    sample_recalls = [
        safe_divide(len(pred & ref), len(ref))
        for pred, ref in zip(pred_sets, ref_sets)
    ]
    sample_f1s = [
        safe_divide(2 * precision * recall, precision + recall)
        for precision, recall in zip(sample_precisions, sample_recalls)
    ]

    micro_tp = sum(len(pred & ref) for pred, ref in zip(pred_sets, ref_sets))
    micro_fp = sum(len(pred - ref) for pred, ref in zip(pred_sets, ref_sets))
    micro_fn = sum(len(ref - pred) for pred, ref in zip(pred_sets, ref_sets))
    micro_precision = safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_divide(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    for label in all_labels:
        label_tp = sum(int(label in pred and label in ref) for pred, ref in zip(pred_sets, ref_sets))
        label_fp = sum(int(label in pred and label not in ref) for pred, ref in zip(pred_sets, ref_sets))
        label_fn = sum(int(label not in pred and label in ref) for pred, ref in zip(pred_sets, ref_sets))
        label_precision = safe_divide(label_tp, label_tp + label_fp)
        label_recall = safe_divide(label_tp, label_tp + label_fn)
        label_f1 = safe_divide(2 * label_precision * label_recall, label_precision + label_recall)
        macro_precisions.append(label_precision)
        macro_recalls.append(label_recall)
        macro_f1s.append(label_f1)

    return {
        "label_exact_match": safe_divide(sum(exact_matches), len(exact_matches)),
        "label_accuracy": safe_divide(sum(exact_matches), len(exact_matches)),
        "label_micro_precision": micro_precision,
        "label_micro_recall": micro_recall,
        "label_micro_f1": micro_f1,
        "label_macro_precision": float(np.mean(macro_precisions)) if macro_precisions else 0.0,
        "label_macro_recall": float(np.mean(macro_recalls)) if macro_recalls else 0.0,
        "label_macro_f1": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
        "label_sample_precision": float(np.mean(sample_precisions)) if sample_precisions else 0.0,
        "label_sample_recall": float(np.mean(sample_recalls)) if sample_recalls else 0.0,
        "label_sample_f1": float(np.mean(sample_f1s)) if sample_f1s else 0.0,
        "label_jaccard": float(np.mean(sample_jaccards)) if sample_jaccards else 0.0,
        "label_count": len(all_labels),
    }


def build_label_metric_result(raw_predictions, references, constrained_predictions=None):
    label_metrics = compute_multilabel_metrics(raw_predictions, references)
    if constrained_predictions is None:
        return label_metrics

    constrained_label_metrics = compute_multilabel_metrics(
        constrained_predictions,
        references,
    )
    return {
        **prefix_metrics(label_metrics, "raw"),
        **prefix_metrics(constrained_label_metrics, "constrained"),
        **constrained_label_metrics,
    }


def mean_bert_f1(bert_f1_scores):
    if bert_f1_scores is None or len(bert_f1_scores) == 0:
        return 0.0
    if torch.is_tensor(bert_f1_scores):
        return bert_f1_scores.mean().item()
    return float(np.mean(bert_f1_scores))


def compute_generation_metrics(
    predictions,
    references,
    bleu_metric,
    rouge_metric,
    meteor_metric,
    bert_scorer=None,
    bert_f1_scores=None,
):
    if not predictions:
        return {
            "rouge-1": 0.0,
            "rouge-L": 0.0,
            "meteor": 0.0,
            "bleu": 0.0,
            "bertscore": 0.0,
        }

    result_bleu = bleu_metric.compute(predictions=predictions, references=references)
    result_rouge = rouge_metric.compute(predictions=predictions, references=references)
    result_meteor = meteor_metric.compute(predictions=predictions, references=references)
    if bert_f1_scores is None:
        if bert_scorer is None:
            raise ValueError("Either bert_scorer or bert_f1_scores must be provided.")
        _, _, bert_f1_scores = bert_scorer.score(predictions, references, verbose=False)

    return {
        "rouge-1": to_float(result_rouge["rouge1"]),
        "rouge-L": to_float(result_rouge["rougeL"]),
        "meteor": to_float(result_meteor["meteor"]),
        "bleu": to_float(result_bleu["score"]),
        "bertscore": mean_bert_f1(bert_f1_scores),
    }


def compute_full_metrics(
    raw_predictions,
    eval_predictions,
    references,
    bleu_metric,
    rouge_metric,
    meteor_metric,
    bert_scorer=None,
    constrained_predictions=None,
    bert_f1_scores=None,
):
    return {
        **build_label_metric_result(raw_predictions, references, constrained_predictions),
        **compute_generation_metrics(
            eval_predictions,
            references,
            bleu_metric,
            rouge_metric,
            meteor_metric,
            bert_scorer,
            bert_f1_scores,
        ),
    }


def select_by_indices(values, indices):
    return [values[idx] for idx in indices]


def select_scores_by_indices(scores, indices):
    if scores is None:
        return None
    if torch.is_tensor(scores):
        return scores[indices]
    return select_by_indices(scores, indices)


def compute_metrics_by_query_mode(
    query_modes,
    raw_predictions,
    eval_predictions,
    references,
    bleu_metric,
    rouge_metric,
    meteor_metric,
    bert_scorer=None,
    constrained_predictions=None,
    bert_f1_scores=None,
):
    metrics_by_mode = {}
    for query_mode in ("image", "text"):
        indices = [idx for idx, mode in enumerate(query_modes) if mode == query_mode]
        if not indices:
            continue
        mode_raw_predictions = select_by_indices(raw_predictions, indices)
        mode_eval_predictions = select_by_indices(eval_predictions, indices)
        mode_references = select_by_indices(references, indices)
        mode_constrained_predictions = (
            select_by_indices(constrained_predictions, indices)
            if constrained_predictions is not None
            else None
        )
        mode_bert_f1_scores = select_scores_by_indices(bert_f1_scores, indices)
        metrics_by_mode[query_mode] = {
            "sample_count": len(indices),
            **compute_full_metrics(
                mode_raw_predictions,
                mode_eval_predictions,
                mode_references,
                bleu_metric,
                rouge_metric,
                meteor_metric,
                bert_scorer,
                mode_constrained_predictions,
                mode_bert_f1_scores,
            ),
        }
    return metrics_by_mode


if __name__ == "__main__":
    args = parse_args()
    validate_ablation_modes(args)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"DSP ablation mode: {args.dsp_ablation_mode}")

    dataset_path = get_dataset_path(args)
    print(f"Loading {args.split} dataset from {dataset_path}")
    personal_dataset = load_from_disk(dataset_path)
    personal_dataset = filter_dataset_for_ablation(args, personal_dataset)
    if args.limit > 0:
        original_size = len(personal_dataset)
        personal_dataset = personal_dataset.select(range(min(args.limit, original_size)))
        print(f"Evaluation dataset size reduced: {original_size} -> {len(personal_dataset)}")
    if args.dump_caption_cache_template:
        dump_caption_cache_template(args, personal_dataset)
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(0)

    if args.mode == "infer":
        if args.ablation_mode != "none":
            print(f"Using ablation inference mode: {args.ablation_mode}")
            print(f"Using plain LLM path: {get_ablation_llm_path(args)}")
            predictions = generate_ablation_predictions(args, personal_dataset)
        else:
            processor = load_or_build_dep_processor(
                base_model_name=args.base_model_name,
                processor_path=args.tokenizer_path,
                prompt_token_count=int(os.getenv("PROMPT_TOKEN_COUNT", "3")),
                label_token_count=get_dataset_label_token_count(personal_dataset),
                save_if_rebuilt=True,
            )
            backend = resolve_backend(args)
            if backend == "vllm" and (
                dataset_has_label_attention(personal_dataset)
                or dataset_has_image_queries(personal_dataset)
            ):
                print(
                    "Multimodal tensors are present; using transformers backend "
                    "because vLLM custom tensor inputs only cover the legacy his_diff_emb path."
                )
                backend = "transformers"
            print(f"Using inference backend: {backend}")
            if backend == "vllm":
                predictions = generate_with_vllm(args, personal_dataset)
            else:
                predictions = generate_with_transformers(args, personal_dataset, processor)
        predictions_path = get_predictions_path(args)
        write_predictions(predictions_path, predictions)
        print(f"Saved predictions to {predictions_path}")
        if args.constrain_labels:
            constrained_predictions = constrain_prediction_labels(
                predictions,
                get_effective_dataset_name(args),
                args.dataset_root,
            )
            constrained_predictions_path = get_constrained_predictions_path(args)
            write_predictions(constrained_predictions_path, constrained_predictions)
            print(f"Saved constrained predictions to {constrained_predictions_path}")

    elif args.mode == "eval":
        references = list(personal_dataset["out_str"])
        predictions_path = get_predictions_path(args)
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = f.read().split("\n---------------------------------\n")
            predictions = predictions[:-1]
        if len(predictions) != len(references):
            raise ValueError(
                "Prediction count does not match evaluation dataset size: "
                f"{len(predictions)} predictions vs {len(references)} references. "
                "Please rerun inference with the same split, dataset path, and limit."
            )

        raw_predictions = [postprocess_output(prediction) for prediction in predictions]
        eval_predictions = raw_predictions
        constrained_predictions = None
        if args.constrain_labels:
            constrained_predictions = constrain_prediction_labels(
                raw_predictions,
                get_effective_dataset_name(args),
                args.dataset_root,
            )
            constrained_predictions_path = get_constrained_predictions_path(args)
            write_predictions(constrained_predictions_path, constrained_predictions)
            print(f"Saved constrained predictions to {constrained_predictions_path}")
            eval_predictions = constrained_predictions

        dataset_name = get_effective_dataset_name(args)
        primary_metric = resolve_primary_metric(dataset_name)
        print(f"Primary evaluation metric for {dataset_name}: {primary_metric.upper()}")

        if primary_metric in {"auroc", "acc"}:
            if primary_metric == "auroc":
                print(
                    "Hateful Memes AUROC uses generated hard-label scores: "
                    "hateful=1, non-hateful=0, ambiguous/invalid=0.5."
                )
            result = compute_task_classification_metrics(
                dataset_name,
                eval_predictions,
                references,
            )
            if "query_mode" in personal_dataset.column_names:
                result["by_query_mode"] = compute_task_metrics_by_query_mode(
                    dataset_name,
                    list(personal_dataset["query_mode"]),
                    eval_predictions,
                    references,
                )
        else:
            bleu_metric = evaluate.load("sacrebleu")
            rouge_metric = evaluate.load("rouge")
            meteor_metric = evaluate.load("meteor")
            bert_scorer = BERTScorer(
                model_type="allenai/led-base-16384",
                lang="en",
                use_fast_tokenizer=False,
            )
            _, _, bert_f1_scores = bert_scorer.score(
                eval_predictions,
                references,
                verbose=False,
            )

            result = compute_full_metrics(
                raw_predictions,
                eval_predictions,
                references,
                bleu_metric,
                rouge_metric,
                meteor_metric,
                bert_scorer,
                constrained_predictions,
                bert_f1_scores,
            )
            if "query_mode" in personal_dataset.column_names:
                result["by_query_mode"] = compute_metrics_by_query_mode(
                    list(personal_dataset["query_mode"]),
                    raw_predictions,
                    eval_predictions,
                    references,
                    bleu_metric,
                    rouge_metric,
                    meteor_metric,
                    bert_scorer,
                    constrained_predictions,
                    bert_f1_scores,
                )
        print(result)

    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)
