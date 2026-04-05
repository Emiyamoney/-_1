from pathlib import Path
import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_paths(config: dict) -> dict:
    root = get_project_root()
    raw_dir = root / config["paths"]["raw_data_dir"]

    return {
        "train_path": raw_dir / config["files"]["train_file"],
        "test_path": raw_dir / config["files"]["test_file"],
        "sample_path": raw_dir / config["files"]["sample_file"],
    }


def load_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def clean_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def load_train_test_sample(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    paths = get_data_paths(config)

    train_df = load_csv(paths["train_path"])
    test_df = load_csv(paths["test_path"])
    sample_df = load_csv(paths["sample_path"])

    train_df = clean_unnamed_columns(train_df)
    test_df = clean_unnamed_columns(test_df)

    return train_df, test_df, sample_df, config