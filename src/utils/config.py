from pathlib import Path
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directories(config: dict):
    root = Path(__file__).resolve().parents[2]

    dirs = [
        config["paths"]["processed_data_dir"],
        config["paths"]["model_dir"],
        config["paths"]["submission_dir"],
        config["paths"]["report_dir"],
        config["paths"]["figure_dir"],
    ]

    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)