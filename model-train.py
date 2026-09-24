import os
import sys
import torch
import warnings
import importlib.util
import inspect
import shutil
import torch.distributed as dist

from datasets import load_from_disk
from transformers import set_seed
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

try:
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
    from utils.multimodal_collator import QwenVLDEPDataCollator
    from utils.tokenizer_utils import load_or_build_dep_processor
except ModuleNotFoundError:
    from promptlearn.DEPMultiRAG.model.personal_model import DEPModel
    from promptlearn.DEPMultiRAG.utils.distributed_utils import (
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
    from promptlearn.DEPMultiRAG.utils.multimodal_collator import QwenVLDEPDataCollator
    from promptlearn.DEPMultiRAG.utils.tokenizer_utils import load_or_build_dep_processor

warnings.filterwarnings("ignore")

set_seed(42)
LOCAL_RANK = resolve_local_rank(sys.argv[1:])
os.environ["LOCAL_RANK"] = str(LOCAL_RANK)
IS_MAIN_PROCESS = LOCAL_RANK == 0
MODEL_DEVICE_MAP = resolve_device_map(LOCAL_RANK, torch.cuda.is_available())
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
print(
    f"[RANK {LOCAL_RANK}] CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')} "
    f"current_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'} "
    f"device_map={MODEL_DEVICE_MAP}",
    flush=True,
)


def pick_attention_impl():
    # Allow manual override: ATTN_IMPL=flash_attention_2|sdpa|eager
    attn_impl = os.getenv("ATTN_IMPL")
    if attn_impl:
        print(f"Using attention implementation from ATTN_IMPL: {attn_impl}")
        return attn_impl

    if importlib.util.find_spec("flash_attn") is not None:
        print("Using attention implementation: flash_attention_2")
        return "flash_attention_2"

    print("flash-attn not found. Falling back to attention implementation: sdpa")
    return "sdpa"


def rank0_print(msg):
    if IS_MAIN_PROCESS:
        print(msg)


def str_to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, keep_checkpoint_paths=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_checkpoint_paths = {
            os.path.abspath(os.path.normpath(path))
            for path in (keep_checkpoint_paths or set())
        }

    def _save_checkpoint(self, model, trial):
        original_save_total_limit = self.args.save_total_limit
        if self.keep_checkpoint_paths:
            self.args.save_total_limit = None
        try:
            super()._save_checkpoint(model, trial)
        finally:
            self.args.save_total_limit = original_save_total_limit

        if self.keep_checkpoint_paths and self.args.should_save:
            self._delete_unkept_checkpoints(trial)

    def _delete_unkept_checkpoints(self, trial):
        run_dir = self._get_output_dir(trial=trial)
        if not os.path.isdir(run_dir):
            return
        checkpoint_paths = [
            os.path.join(run_dir, name)
            for name in os.listdir(run_dir)
            if name.startswith("checkpoint-")
            and os.path.isdir(os.path.join(run_dir, name))
        ]
        for checkpoint_path in filter_checkpoints_for_retention(
            checkpoint_paths,
            self.keep_checkpoint_paths,
        ):
            if os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path, ignore_errors=True)
                rank0_print(f"Deleted checkpoint not in KEEP_CHECKPOINTS: {checkpoint_path}")

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

llm_model_name = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
tokenizer_path = os.getenv("TOKENIZER_PATH", "output/DEP-tokenizer")
resume_checkpoint = os.getenv("RESUME_FROM_CHECKPOINT", "").strip()
resume_model_file = os.path.join(resume_checkpoint, "model.safetensors")
model_source = llm_model_name
if resume_checkpoint and os.path.exists(resume_model_file):
    model_source = resume_checkpoint
    rank0_print(
        "Resume checkpoint contains model weights but may miss trainer state; "
        f"loading model from: {resume_checkpoint}"
    )
attn_impl = pick_attention_impl()
max_train_seq_len = int(os.getenv("TRAIN_MAX_LENGTH", "2048"))
deepspeed_config = os.getenv("DEEPSPEED_CONFIG", "deepspeed/ds_z2_frozen_config.json")
use_gc = str_to_bool(os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1"))
use_reentrant_gc = resolve_gradient_checkpointing_reentrant(default=False)
trust_remote_code = str_to_bool(os.getenv("TRUST_REMOTE_CODE", "0"))
hf_local_files_only = str_to_bool(os.getenv("HF_LOCAL_FILES_ONLY", "0"))
rag_embed_size = int(os.getenv("RAG_EMBED_SIZE", "768"))
sparse_hidden_size = int(os.getenv("SPARSE_HIDDEN_SIZE", "512"))
prompt_token_count = int(os.getenv("PROMPT_TOKEN_COUNT", "3"))
sae_loss_weight = float(os.getenv("SAE_LOSS_WEIGHT", "0.1"))
sae_sparse_weight = float(os.getenv("SAE_SPARSE_WEIGHT", "0.001"))
label_loss_weight = float(os.getenv("LABEL_LOSS_WEIGHT", "0.8"))
train_data_dir = os.getenv("TRAIN_DATA_DIR", "data/dataset_train")
dataset_root = os.getenv("RAGPT_DATA_ROOT", "dataset")
dataset_name = os.getenv("RALAP_DATASET_NAME", "")
train_subset_divisor = int(os.getenv("TRAIN_SUBSET_DIVISOR", "1"))
num_train_epochs = float(os.getenv("NUM_TRAIN_EPOCHS", "5"))
output_dir = resolve_output_dir(default="output")
save_strategy = os.getenv("SAVE_STRATEGY", "epoch")
save_steps = int(os.getenv("SAVE_STEPS", "500"))
save_only_model = resolve_save_only_model(default=True)
save_total_limit = resolve_save_total_limit(default=1)
keep_checkpoint_paths = resolve_checkpoint_retention_paths(output_dir=output_dir)
effective_save_total_limit = None if keep_checkpoint_paths else save_total_limit
report_to = os.getenv("REPORT_TO", "wandb")
label_attn_size = int(os.getenv("LABEL_ATTN_SIZE", "256"))
label_attn_loss_weight = float(os.getenv("LABEL_ATTN_LOSS_WEIGHT", "0.05"))
gradient_accumulation_steps = resolve_gradient_accumulation_steps(default=16)
training_args = Seq2SeqTrainingArguments(
    num_train_epochs=num_train_epochs,
    output_dir=output_dir,
    logging_steps=10,
    save_strategy=save_strategy,
    save_steps=save_steps,
    save_only_model=save_only_model,
    save_total_limit=effective_save_total_limit,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim="adamw_torch",
    learning_rate=1e-5,
    weight_decay=0.025,
    warmup_ratio=0.01,
    bf16=True,
    gradient_checkpointing=use_gc,
    gradient_checkpointing_kwargs={"use_reentrant": use_reentrant_gc},
    deepspeed=deepspeed_config,
    report_to=report_to,
    run_name="DEP",
    remove_unused_columns=False,
)
personal_dataset = load_from_disk(train_data_dir)
label_token_count = 0
if len(personal_dataset) > 0 and "target_label_mask" in personal_dataset.column_names:
    label_token_count = len(personal_dataset[0]["target_label_mask"])
processor = load_or_build_dep_processor(
    base_model_name=llm_model_name,
    processor_path=tokenizer_path,
    prompt_token_count=prompt_token_count,
    label_token_count=label_token_count,
    save_if_rebuilt=True,
)
llm_tokenizer = processor.tokenizer
personal_model = DEPModel.from_pretrained(
    model_source,
    device_map=MODEL_DEVICE_MAP,
    torch_dtype=torch.bfloat16,
    trust_remote_code=trust_remote_code,
    local_files_only=hf_local_files_only,
    attn_implementation=attn_impl,
    training=True,
    tokenizer=llm_tokenizer,
    rag_embed_size=rag_embed_size,
    sparse_hidden_size=sparse_hidden_size,
    prompt_token_count=prompt_token_count,
    label_token_count=label_token_count,
    label_attn_size=label_attn_size,
    label_attn_loss_weight=label_attn_loss_weight,
    sae_loss_weight=sae_loss_weight,
    sae_sparse_weight=sae_sparse_weight,
    label_loss_weight=label_loss_weight,
)
personal_model.resize_token_embeddings(len(llm_tokenizer), mean_resizing=False)
personal_model.freeze_non_dep_parameters()
personal_model.config.use_cache = False
if use_gc:
    personal_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant_gc}
    )

rank0_print(f"Using DeepSpeed config: {deepspeed_config}")
rank0_print(f"Training max sequence length (batch truncation): {max_train_seq_len}")
rank0_print(f"Training data directory: {train_data_dir}")
rank0_print(
    f"RAG prompt config: embed_size={rag_embed_size}, "
    f"hidden_size={sparse_hidden_size}, prompt_token_count={prompt_token_count}, "
    f"sae_loss_weight={sae_loss_weight}, sae_sparse_weight={sae_sparse_weight}, "
    f"label_loss_weight={label_loss_weight}, label_token_count={label_token_count}, "
    f"label_attn_size={label_attn_size}, label_attn_loss_weight={label_attn_loss_weight}"
)
rank0_print(
    "Gradient checkpointing: "
    f"{'on' if use_gc else 'off'} (use_reentrant={use_reentrant_gc})"
)
rank0_print(f"Trust remote code: {trust_remote_code}")
rank0_print(f"HF local files only: {hf_local_files_only}")
rank0_print(
    f"Trainer control: num_train_epochs={num_train_epochs}, "
    f"output_dir={output_dir}, save_strategy={save_strategy}, "
    f"save_steps={save_steps}, save_only_model={save_only_model}, "
    f"save_total_limit={effective_save_total_limit}, "
    f"keep_checkpoints={sorted(keep_checkpoint_paths)}, report_to={report_to}"
)
rank0_print(str(personal_model))
if IS_MAIN_PROCESS:
    print_trainable_parameters(personal_model)

original_train_size = len(personal_dataset)
if train_subset_divisor > 1:
    subset_size = max(1, original_train_size // train_subset_divisor)
    personal_dataset = personal_dataset.shuffle(seed=42).select(range(subset_size))
    rank0_print(
        f"Train dataset size reduced: {original_train_size} -> {subset_size} "
        f"(1/{train_subset_divisor})"
    )
else:
    rank0_print(f"Train dataset size: {original_train_size}")
trainer_kwargs = {
    "model": personal_model,
    "args": training_args,
    "train_dataset": personal_dataset,
    "data_collator": QwenVLDEPDataCollator(
        processor=processor,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        max_length=max_train_seq_len,
        training=True,
    ),
    "keep_checkpoint_paths": keep_checkpoint_paths,
}
trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
if "processing_class" in trainer_init_params:
    trainer_kwargs["processing_class"] = processor
elif "tokenizer" in trainer_init_params:
    trainer_kwargs["tokenizer"] = llm_tokenizer

trainer = CustomTrainer(**trainer_kwargs)

rank0_print("train start")
if resume_checkpoint:
    trainer_state_file = os.path.join(resume_checkpoint, "trainer_state.json")
    if os.path.exists(trainer_state_file):
        rank0_print(f"Resuming full trainer state from: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        rank0_print(
            "trainer_state.json not found in resume checkpoint. "
            "Will continue training from loaded model weights only."
        )
        trainer.train()
else:
    trainer.train()
rank0_print("train done")
if dist.is_initialized():
    dist.destroy_process_group()
sys.exit(0)
