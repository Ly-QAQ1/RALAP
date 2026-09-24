import os


def _normalize_checkpoint_path(path, output_dir):
    path = os.path.expanduser(str(path).strip())
    if not path:
        return None
    if not os.path.isabs(path) and os.path.dirname(os.path.normpath(path)) == "":
        path = os.path.join(output_dir, path)
    return os.path.abspath(os.path.normpath(path))


def resolve_local_rank(argv=None, environ=None):
    argv = [] if argv is None else list(argv)
    environ = os.environ if environ is None else environ

    for idx, arg in enumerate(argv):
        if arg == "--local_rank" and idx + 1 < len(argv):
            return int(argv[idx + 1])
        if arg.startswith("--local_rank="):
            return int(arg.split("=", 1)[1])

    return int(environ.get("LOCAL_RANK", environ.get("SLURM_LOCALID", "0")))


def resolve_device_map(local_rank, cuda_available=True):
    if not cuda_available:
        return None
    return {"": f"cuda:{int(local_rank)}"}


def resolve_gradient_accumulation_steps(environ=None, default=16):
    environ = os.environ if environ is None else environ
    return int(environ.get("GRADIENT_ACCUMULATION_STEPS", default))


def resolve_gradient_checkpointing_reentrant(environ=None, default=False):
    environ = os.environ if environ is None else environ
    value = environ.get("GC_USE_REENTRANT")
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def resolve_output_dir(environ=None, default="output"):
    environ = os.environ if environ is None else environ
    return environ.get("OUTPUT_DIR", default)


def resolve_save_only_model(environ=None, default=True):
    environ = os.environ if environ is None else environ
    value = environ.get("SAVE_ONLY_MODEL")
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def resolve_save_total_limit(environ=None, default=1):
    environ = os.environ if environ is None else environ
    value = environ.get("SAVE_TOTAL_LIMIT")
    if value is None:
        return default
    value = str(value).strip()
    if value.lower() in ("", "none", "null"):
        return None
    return int(value)


def resolve_checkpoint_retention_paths(environ=None, output_dir="output"):
    environ = os.environ if environ is None else environ
    value = environ.get("KEEP_CHECKPOINTS", "")
    paths = set()
    for item in str(value).split(","):
        path = _normalize_checkpoint_path(item, output_dir)
        if path is not None:
            paths.add(path)
    return paths


def filter_checkpoints_for_retention(checkpoints, keep_paths):
    normalized_keep_paths = {
        os.path.abspath(os.path.normpath(path))
        for path in keep_paths
    }
    to_delete = []
    for checkpoint in checkpoints:
        normalized_checkpoint = os.path.abspath(os.path.normpath(checkpoint))
        if normalized_checkpoint not in normalized_keep_paths:
            to_delete.append(normalized_checkpoint)
    return to_delete
