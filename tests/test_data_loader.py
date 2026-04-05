import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.loader import load_train_test_sample


def test_load_data():
    train_df, test_df, sample_df, config = load_train_test_sample()
    assert train_df is not None
    assert test_df is not None
    assert sample_df is not None
    assert len(train_df) > 0