import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.preprocessor import CreditDataPreprocessor


def test_preprocessor():
    df = pd.DataFrame({
        "age": [20, -1, 120],
        "MonthlyIncome": [5000, None, -100],
        "DebtRatio": [0.5, 1000, -1],
    })

    processor = CreditDataPreprocessor()
    out = processor.fit_transform(df)

    assert out["age"].min() >= 18
    assert out["age"].max() <= 100
    assert out["MonthlyIncome"].min() >= 0