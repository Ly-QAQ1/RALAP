import os
import sys
import importlib.util
import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.personal_model import DEPModel
from utils.distributed_utils import (
    filter_checkpoints_for_retention,
    resolve_checkpoint_retention_paths,
    resolve_device_map,
    resolve_gradient_accumulation_steps,
    resolve_gradient_checkpointing_reentrant,
    resolve_local_rank,
    resolve_output_dir,
    resolve_save_only_model,
    resolve_save_total_limit,
)
from utils.tokenizer_utils import _apply_qwenvl_pixel_budget


def _load_create_dataset_module():
    module_path = Path(REPO_ROOT) / "create-dataset.py"
    spec = importlib.util.spec_from_file_location("create_dataset_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model_eval_module():
    module_path = Path(REPO_ROOT) / "model-eval.py"
    spec = importlib.util.spec_from_file_location("model_eval_for_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeImageProcessor:
    def __init__(self):
        self.size = {"shortest_edge": 3136, "longest_edge": 12845056}


class _FakeProcessor:
    def __init__(self):
        self.image_processor = _FakeImageProcessor()
        self.video_processor = _FakeImageProcessor()


class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


class _RecordingLMHead(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(vocab_size, hidden_size))
        self.calls = []

    def forward(self, hidden_states):
        self.calls.append(tuple(hidden_states.shape))
        return torch.zeros(
            hidden_states.size(0),
            self.weight.size(0),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )


def test_pixel_budget_updates_qwenvl_image_and_video_processors():
    processor = _FakeProcessor()

    _apply_qwenvl_pixel_budget(processor, min_pixels=3136, max_pixels=200704)

    assert processor.image_processor.size["shortest_edge"] == 3136
    assert processor.image_processor.size["longest_edge"] == 200704
    assert processor.video_processor.size["shortest_edge"] == 3136
    assert processor.video_processor.size["longest_edge"] == 200704


def test_hidden_loss_projects_only_supervised_tokens():
    model = DEPModel.__new__(DEPModel)
    torch.nn.Module.__init__(model)
    model.label_loss_weight = 0.8
    model._explanation_pattern_ids = []
    model.lm_head = _RecordingLMHead(hidden_size=4, vocab_size=11)

    hidden_states = torch.randn(1, 6, 4, dtype=torch.bfloat16)
    labels = torch.tensor([[-100, -100, -100, 3, 4, 5]])

    loss = model._compute_label_explanation_loss_from_hidden(hidden_states, labels)

    assert torch.isfinite(loss)
    assert model.lm_head.calls == [(3, 4)]


def test_resolve_local_rank_accepts_deepspeed_cli_argument():
    assert resolve_local_rank(["model-train.py", "--local_rank=1"], {}) == 1
    assert resolve_local_rank(["model-train.py", "--local_rank", "2"], {}) == 2
    assert resolve_local_rank(["model-train.py"], {"LOCAL_RANK": "3"}) == 3


def test_device_map_targets_current_local_rank():
    assert resolve_device_map(1, cuda_available=True) == {"": "cuda:1"}
    assert resolve_device_map(1, cuda_available=False) is None


def test_training_config_helpers_default_to_stable_deepspeed_values():
    assert resolve_gradient_accumulation_steps({}, default=16) == 16
    assert resolve_gradient_accumulation_steps({"GRADIENT_ACCUMULATION_STEPS": "4"}, default=16) == 4
    assert resolve_gradient_checkpointing_reentrant({}, default=False) is False
    assert resolve_gradient_checkpointing_reentrant({"GC_USE_REENTRANT": "1"}, default=False) is True


def test_checkpoint_config_helpers_default_to_compact_model_only_saves():
    assert resolve_output_dir({}) == "output"
    assert resolve_output_dir({"OUTPUT_DIR": "/tmp/dep-output"}) == "/tmp/dep-output"
    assert resolve_save_only_model({}, default=True) is True
    assert resolve_save_only_model({"SAVE_ONLY_MODEL": "0"}, default=True) is False
    assert resolve_save_total_limit({}, default=1) == 1
    assert resolve_save_total_limit({"SAVE_TOTAL_LIMIT": "3"}, default=1) == 3
    assert resolve_save_total_limit({"SAVE_TOTAL_LIMIT": "none"}, default=1) is None


def test_checkpoint_retention_paths_are_normalized_against_output_dir():
    paths = resolve_checkpoint_retention_paths(
        {"KEEP_CHECKPOINTS": "checkpoint-4860, output/checkpoint-9720"},
        output_dir="output",
    )

    assert paths == {
        os.path.abspath(os.path.normpath("output/checkpoint-4860")),
        os.path.abspath(os.path.normpath("output/checkpoint-9720")),
    }


def test_checkpoint_retention_deletes_every_checkpoint_not_explicitly_kept():
    checkpoints = [
        "output/checkpoint-972",
        "output/checkpoint-1944",
        "output/checkpoint-4860",
        "output/checkpoint-5832",
    ]
    keep_paths = resolve_checkpoint_retention_paths(
        {"KEEP_CHECKPOINTS": "output/checkpoint-4860"},
        output_dir="output",
    )

    to_delete = filter_checkpoints_for_retention(
        checkpoints=checkpoints,
        keep_paths=keep_paths,
    )

    assert to_delete == [
        os.path.abspath(os.path.normpath("output/checkpoint-972")),
        os.path.abspath(os.path.normpath("output/checkpoint-1944")),
        os.path.abspath(os.path.normpath("output/checkpoint-5832")),
    ]


def test_create_dataset_load_split_runs_retrieval_preparation_when_columns_missing():
    module = _load_create_dataset_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "food101"
        dataset_dir.mkdir(parents=True)
        split_path = dataset_dir / "train.pkl"
        pd.DataFrame(
            {
                "item_id": ["sample-1"],
                "text": ["a plate of dumplings"],
                "label": [0],
            }
        ).to_pickle(split_path)

        module.DATASET_ROOT = Path(tmpdir)
        module.DATASET_NAME = "food101"
        calls = []

        def fake_prepare_retrieval_data_once():
            calls.append("prepared")
            df = pd.read_pickle(split_path)
            df["t2t_id_list"] = [["sample-2"]]
            df["t2t_sims_list"] = [[0.9]]
            df["t2t_label_list"] = [[1]]
            df["i2i_id_list"] = [["sample-3"]]
            df["i2i_sims_list"] = [[0.8]]
            df["i2i_label_list"] = [[2]]
            df.to_pickle(split_path)

        module.prepare_retrieval_data_once = fake_prepare_retrieval_data_once

        loaded = module.load_split("train")

    assert calls == ["prepared"]
    assert set(module.REQUIRED_RETRIEVAL_COLUMNS).issubset(loaded.columns)


def test_create_dataset_load_split_runs_preparation_when_split_is_missing():
    module = _load_create_dataset_module()

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "hatememes"
        split_path = dataset_dir / "train.pkl"
        module.DATASET_ROOT = Path(tmpdir)
        module.DATASET_NAME = "hatememes"
        calls = []

        def fake_prepare_retrieval_data_once():
            calls.append("prepared")
            dataset_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "item_id": ["01235"],
                    "text": ["example meme text"],
                    "label": [1],
                    "t2t_id_list": [["01236"]],
                    "t2t_sims_list": [[0.9]],
                    "t2t_label_list": [[0]],
                    "i2i_id_list": [["01243"]],
                    "i2i_sims_list": [[0.8]],
                    "i2i_label_list": [[1]],
                }
            ).to_pickle(split_path)

        module.prepare_retrieval_data_once = fake_prepare_retrieval_data_once

        loaded = module.load_split("train")

    assert calls == ["prepared"]
    assert loaded["item_id"].tolist() == ["01235"]


def test_run_create_prefers_dep_python_when_no_explicit_python_bin_is_set():
    script_text = (Path(REPO_ROOT) / "run-create.sh").read_text(encoding="utf-8")

    assert "/home/dell/anaconda3/envs/dep/bin/python" in script_text
    assert "CONDA_DEFAULT_ENV" in script_text
    assert "${PYTHON_BIN:-}" in script_text


def test_run_create_maps_gpu_env_to_cuda_visible_devices():
    script_text = (Path(REPO_ROOT) / "run-create.sh").read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES" in script_text
    assert "${GPU}" in script_text


def test_mcr_similarity_uses_matrix_cosine_without_broadcasting():
    import core_toolsRAGPT

    mcr = core_toolsRAGPT.MCR.__new__(core_toolsRAGPT.MCR)
    mcr.batch_size = 2
    mcr.top_k = 3

    query_vectors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    memory_bank = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
            [-0.6, 0.8],
        ]
    )
    memory_bank_id = ["q0", "q1", "m2", "m3"]
    memory_bank_label = [0, 1, 2, 3]

    original_cosine_similarity = core_toolsRAGPT.F.cosine_similarity

    def fail_if_broadcast_cosine_is_used(*args, **kwargs):
        raise AssertionError("broadcast cosine_similarity should not be used")

    core_toolsRAGPT.F.cosine_similarity = fail_if_broadcast_cosine_is_used
    try:
        ids, sims, labels = mcr._compute_similarity_in_batches(
            query_vectors,
            memory_bank,
            memory_bank_id,
            memory_bank_label,
        )
    finally:
        core_toolsRAGPT.F.cosine_similarity = original_cosine_similarity

    assert ids == [["m2", "q1"], ["m3", "m2"]]
    assert labels == [[2, 1], [3, 2]]
    assert torch.allclose(torch.tensor(sims), torch.tensor([[0.8, 0.0], [0.8, 0.6]]))


def test_primary_metric_is_dataset_specific_and_preserves_mmimdb():
    module = _load_model_eval_module()

    assert module.resolve_primary_metric("hatememes") == "auroc"
    assert module.resolve_primary_metric("food101") == "acc"
    assert module.resolve_primary_metric("mmimdb") == "multilabel"


def test_hatememes_auroc_uses_generated_binary_labels():
    module = _load_model_eval_module()
    references = [
        "Label: hateful",
        "Label: hateful",
        "Label: non-hateful",
        "Label: non-hateful",
    ]
    predictions = [
        "Label: hateful",
        "Label: hateful",
        "Label: non-hateful",
        "Label: non-hateful",
    ]

    metrics = module.compute_task_classification_metrics(
        "hatememes",
        predictions,
        references,
    )

    assert metrics == {"auroc": 1.0}


def test_food101_acc_uses_exact_single_label_match():
    module = _load_model_eval_module()
    references = [
        "Label: apple pie",
        "Label: sushi",
        "Label: pizza",
    ]
    predictions = [
        "Label: apple pie",
        "Label: sushi, pizza",
        "Label: ramen",
    ]

    metrics = module.compute_task_classification_metrics(
        "food101",
        predictions,
        references,
    )

    assert metrics == {"acc": 1.0 / 3.0}


def test_dsp_ablation_modes_select_expected_components():
    module = _load_model_eval_module()

    assert module.resolve_dsp_components("full") == (True, True)
    assert module.resolve_dsp_components("sae_only") == (True, False)
    assert module.resolve_dsp_components("label_only") == (False, True)
    assert module.resolve_dsp_components("no_dsp") == (False, False)


def test_dsp_prompt_ablation_removes_only_disabled_soft_prompt_blocks():
    module = _load_model_eval_module()
    prompt = (
        "[Text Retrieved Label Candidates]: Drama, Comedy\n"
        "[Target Text]: an original plot description\n"
        "[Self Prompt]: <his_token_start>[HIS_TOKEN_0]<his_token_end>\n"
        "[Difference Prompt]: <diff_token_start>[DIFF_TOKEN_0]<diff_token_end>\n"
        "<label_attn_start>\n"
        "[Label Attention Table]\n"
        "| Label | Attention |\n"
        "| Drama | [LAB_ATTN_0] |\n"
        "<label_attn_end>\n"
    )

    sae_only = module.apply_dsp_prompt_ablation(prompt, "sae_only")
    assert "[HIS_TOKEN_0]" in sae_only
    assert "[DIFF_TOKEN_0]" in sae_only
    assert "[LAB_ATTN_0]" not in sae_only

    label_only = module.apply_dsp_prompt_ablation(prompt, "label_only")
    assert "[HIS_TOKEN_0]" not in label_only
    assert "[DIFF_TOKEN_0]" not in label_only
    assert "[LAB_ATTN_0]" in label_only

    no_dsp = module.apply_dsp_prompt_ablation(prompt, "no_dsp")
    assert "[HIS_TOKEN_0]" not in no_dsp
    assert "[DIFF_TOKEN_0]" not in no_dsp
    assert "[LAB_ATTN_0]" not in no_dsp
    assert "[Target Text]: an original plot description" in no_dsp
    assert "[Text Retrieved Label Candidates]: Drama, Comedy" in no_dsp


def test_dsp_generation_kwargs_omit_disabled_component_tensors():
    module = _load_model_eval_module()
    his_diff = torch.ones(1, 4, 3)
    label_inputs = {
        "label_attn_target_emb": torch.ones(1, 3),
        "label_attn_retrieved_emb": torch.ones(1, 2, 3),
        "label_attn_label_mask": torch.ones(1, 2, 4),
        "label_attn_valid_mask": torch.ones(1, 2),
    }

    sae_only = module.add_dsp_generation_inputs({}, "sae_only", his_diff, label_inputs)
    assert sae_only["use_sae_dsp"] is True
    assert sae_only["use_label_dsp"] is False
    assert sae_only["his_diff_emb"] is his_diff
    assert not set(label_inputs).intersection(sae_only)

    label_only = module.add_dsp_generation_inputs({}, "label_only", his_diff, label_inputs)
    assert label_only["use_sae_dsp"] is False
    assert label_only["use_label_dsp"] is True
    assert "his_diff_emb" not in label_only
    assert all(label_only[key] is value for key, value in label_inputs.items())

    no_dsp = module.add_dsp_generation_inputs({}, "no_dsp", his_diff, label_inputs)
    assert no_dsp == {"use_sae_dsp": False, "use_label_dsp": False}


def test_dsp_ablation_prediction_paths_are_isolated_and_full_is_compatible():
    module = _load_model_eval_module()

    def make_args(mode):
        return SimpleNamespace(
            predictions_path="",
            constrained_predictions_path="",
            ablation_mode="none",
            dsp_ablation_mode=mode,
            model_path="output/checkpoint-10",
            split="test",
        )

    assert module.get_predictions_path(make_args("full")).endswith("predictions_test.txt")
    assert module.get_predictions_path(make_args("sae_only")).endswith(
        "predictions_test_dsp_sae_only.txt"
    )
    assert module.get_predictions_path(make_args("label_only")).endswith(
        "predictions_test_dsp_label_only.txt"
    )
    assert module.get_predictions_path(make_args("no_dsp")).endswith(
        "predictions_test_dsp_no_dsp.txt"
    )


def test_dsp_ablation_rejects_combining_with_legacy_ablation():
    module = _load_model_eval_module()
    args = SimpleNamespace(ablation_mode="text_only", dsp_ablation_mode="sae_only")

    try:
        module.validate_ablation_modes(args)
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("Expected incompatible ablation modes to raise ValueError")


def test_dep_forward_dsp_switches_default_to_full_model_behavior():
    signature = inspect.signature(DEPModel.forward)

    assert signature.parameters["use_sae_dsp"].default is True
    assert signature.parameters["use_label_dsp"].default is True


def test_run_eval_exposes_dsp_mode_without_changing_temperature_default():
    script_text = (Path(REPO_ROOT) / "run-eval.sh").read_text(encoding="utf-8")

    assert 'TEMPERATURE="${TEMPERATURE:-0.1}"' in script_text
    assert 'DSP_ABLATION_MODE="${DSP_ABLATION_MODE:-full}"' in script_text
    assert '--dsp_ablation_mode "$DSP_ABLATION_MODE"' in script_text


def test_soft_prompt_replacement_preserves_gradient_for_checkpointed_inputs():
    model = DEPModel.__new__(DEPModel)
    torch.nn.Module.__init__(model)
    model.prompt_token_count = 2
    model.llm_tokenizer = _FakeTokenizer({"[HIS_TOKEN_0]": 10, "[HIS_TOKEN_1]": 11})

    inputs_embs = torch.zeros(1, 4, 3, requires_grad=True)
    input_ids = torch.tensor([[99, 10, 88, 11]])
    prompt_emb = torch.arange(6, dtype=torch.float32).view(1, 2, 3).requires_grad_()

    updated = model._replace_soft_prompt_tokens(inputs_embs, input_ids, prompt_emb, "HIS")

    assert torch.equal(updated[0, 1], prompt_emb[0, 0])
    assert torch.equal(updated[0, 3], prompt_emb[0, 1])
    updated.sum().backward()
    assert prompt_emb.grad is not None
    assert torch.count_nonzero(prompt_emb.grad) == prompt_emb.numel()


def test_label_attention_replacement_preserves_gradient_for_checkpointed_inputs():
    model = DEPModel.__new__(DEPModel)
    torch.nn.Module.__init__(model)
    model.label_token_count = 1
    model.llm_tokenizer = _FakeTokenizer({"[LAB_ATTN_0]": 20})

    inputs_embs = torch.ones(1, 3, 2, requires_grad=True)
    input_ids = torch.tensor([[99, 20, 88]])
    label_prompt_emb = torch.full((1, 1, 2), 2.0, requires_grad=True)
    gate = torch.tensor(0.25, requires_grad=True)

    updated = model._replace_label_attention_tokens(
        inputs_embs,
        input_ids,
        label_prompt_emb,
        gate,
    )

    assert torch.allclose(updated[0, 1], torch.full((2,), 1.5))
    updated.sum().backward()
    assert label_prompt_emb.grad is not None
    assert torch.count_nonzero(label_prompt_emb.grad) == label_prompt_emb.numel()
    assert gate.grad is not None


if __name__ == "__main__":
    test_pixel_budget_updates_qwenvl_image_and_video_processors()
    test_hidden_loss_projects_only_supervised_tokens()
    test_resolve_local_rank_accepts_deepspeed_cli_argument()
    test_device_map_targets_current_local_rank()
    test_training_config_helpers_default_to_stable_deepspeed_values()
    test_checkpoint_config_helpers_default_to_compact_model_only_saves()
    test_checkpoint_retention_paths_are_normalized_against_output_dir()
    test_checkpoint_retention_deletes_every_checkpoint_not_explicitly_kept()
    test_create_dataset_load_split_runs_retrieval_preparation_when_columns_missing()
    test_create_dataset_load_split_runs_preparation_when_split_is_missing()
    test_run_create_prefers_dep_python_when_no_explicit_python_bin_is_set()
    test_run_create_maps_gpu_env_to_cuda_visible_devices()
    test_mcr_similarity_uses_matrix_cosine_without_broadcasting()
    test_primary_metric_is_dataset_specific_and_preserves_mmimdb()
    test_hatememes_auroc_uses_generated_binary_labels()
    test_food101_acc_uses_exact_single_label_match()
    test_dsp_ablation_modes_select_expected_components()
    test_dsp_prompt_ablation_removes_only_disabled_soft_prompt_blocks()
    test_dsp_generation_kwargs_omit_disabled_component_tensors()
    test_dsp_ablation_prediction_paths_are_isolated_and_full_is_compatible()
    test_dsp_ablation_rejects_combining_with_legacy_ablation()
    test_dep_forward_dsp_switches_default_to_full_model_behavior()
    test_run_eval_exposes_dsp_mode_without_changing_temperature_default()
    test_soft_prompt_replacement_preserves_gradient_for_checkpointed_inputs()
    test_label_attention_replacement_preserves_gradient_for_checkpointed_inputs()
