import sys
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.loader import load_train_test_sample
from src.utils.config import ensure_directories
from src.utils.logger import get_logger


def assign_credit_grade(probabilities, bins, labels):
    grades = pd.cut(
        probabilities,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return grades.astype(str)


def predict():
    logger = get_logger()
    train_df, test_df, sample_df, config = load_train_test_sample()
    ensure_directories(config)

    model_path = ROOT / config["paths"]["model_dir"] / config["files"]["model_file"]
    submission_dir = ROOT / config["paths"]["submission_dir"]
    report_dir = ROOT / config["paths"]["report_dir"]

    submission_path = submission_dir / config["files"]["submission_file"]
    report_path = report_dir / config["files"]["prediction_report_file"]

    bundle = joblib.load(model_path)

    preprocessor = bundle["preprocessor"]
    engineer = bundle["engineer"]
    selector = bundle["selector"]
    model = bundle["model"]

    # 测试集里目标列为空，删掉再预测
    if config["target"]["name"] in test_df.columns:
        X_test = test_df.drop(columns=[config["target"]["name"]])
    else:
        X_test = test_df.copy()

    X_test_p = preprocessor.transform(X_test)
    X_test_f = engineer.transform(X_test_p)
    X_test_s = selector.transform(X_test_f)

    test_prob = model.predict_proba(X_test_s)

    bins = config["credit_grade"]["bins"]
    labels = config["credit_grade"]["labels"]
    grades = assign_credit_grade(test_prob, bins, labels)

    # 输出 Kaggle 风格提交文件
    if "Id" in sample_df.columns:
        submission = pd.DataFrame({
            "Id": sample_df["Id"],
            "Probability": test_prob
        })
    else:
        submission = pd.DataFrame({
            "Id": range(1, len(test_prob) + 1),
            "Probability": test_prob
        })

    submission.to_csv(submission_path, index=False)

    # 输出带信用等级的结果文件
    prediction_report = submission.copy()
    prediction_report["CreditLevel"] = grades
    prediction_report.to_csv(report_path, index=False)

    logger.info(f"预测完成，submission已保存到: {submission_path}")
    logger.info(f"信用等级结果已保存到: {report_path}")

    return prediction_report.head()


if __name__ == "__main__":
    print(predict())