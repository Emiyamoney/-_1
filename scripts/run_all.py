import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.train import train
from scripts.predict import predict
from scripts.evaluate import evaluate


def main():
    print("=" * 60)
    print("Step 1: Train model")
    print("=" * 60)
    train()

    print("\n" + "=" * 60)
    print("Step 2: Predict credit risk")
    print("=" * 60)
    preview = predict()
    print(preview)

    print("\n" + "=" * 60)
    print("Step 3: Evaluate model")
    print("=" * 60)
    evaluate()

    print("\n全部完成。")
    print("输出文件位置：")
    print("1. models/credit_risk_model.joblib")
    print("2. results/submissions/submission.csv")
    print("3. results/reports/test_predictions_with_grade.csv")
    print("4. results/reports/metrics.json")


if __name__ == "__main__":
    main()