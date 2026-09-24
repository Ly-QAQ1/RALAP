from types import SimpleNamespace
from pathlib import Path
import warnings

import pandas as pd
import yaml

try:
    from core_toolsRAGPT import (
        init_data_mmimdb,
        init_data_hatememes,
        init_data_food101,
        MemoryBankGenerator,
        MCR,
    )
except ModuleNotFoundError as exc:
    if exc.name != "core_toolsRAGPT":
        raise
    from promptlearn.DEPMultiRAG.core_toolsRAGPT import (
        init_data_mmimdb,
        init_data_hatememes,
        init_data_food101,
        MemoryBankGenerator,
        MCR,
    )

warnings.filterwarnings("ignore")


def dict_to_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{key: dict_to_namespace(value) for key, value in data.items()})
    if isinstance(data, list):
        return [dict_to_namespace(value) for value in data]
    return data


def load_cfg(config_path="config/config.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            "Please copy RAGPT config first, e.g. "
            "`mkdir -p config && cp ../RAGPT-master/src/config/ragpt_config.yaml config/config.yaml`."
        )
    with open(path, "r", encoding="utf-8") as f:
        return dict_to_namespace(yaml.safe_load(f))


def main(cfg=None):
    if cfg is None:
        cfg = load_cfg()

    pd.set_option("future.no_silent_downcasting", True)
    print("==> Data Initialization start.")
    if cfg.data_para.dataset_name == "mmimdb":
        init_data_mmimdb()
    elif cfg.data_para.dataset_name == "hatememes":
        init_data_hatememes()
    elif cfg.data_para.dataset_name == "food101":
        init_data_food101()
    else:
        raise ValueError(f"Unsupported dataset: {cfg.data_para.dataset_name}")

    print("==> Data Initialization finished.")
    print("==> Memory Bank Generation start.")
    memory_bank_generator = MemoryBankGenerator(cfg)
    memory_bank_generator.run()
    print("==> Memory Bank Generation finished.")
    print("==> Retrieval start.")
    mcr = MCR(cfg)
    mcr.run()
    print("==> Retrieval finished.")


if __name__ == "__main__":
    main()
