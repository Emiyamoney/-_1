import sys
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.loader import load_train_test_sample
from src.data.preprocessor import CreditDataPreprocessor
from src.features.engineering import FeatureEngineer
from src.features.selection import FeatureSelector
from src.models.baseline import CreditRiskEnsembleModel
from src.evaluation.metrics import classification_metrics, save_metrics
from src.evaluation.explain import save_feature_importance_plot
from src.utils.config import ensure_directories
from src.utils.logger import get_logger


def train():
    logger = get_logger()
    train_df, test_df, sample_df, config = load_train_test_sample()
    ensure_directories(config)

    target_col = config["target"]["name"]

    logger.info("开始训练模型...")

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["project"]["random_state"],
        stratify=y,
    )

    preprocessor = CreditDataPreprocessor()
    engineer = FeatureEngineer()
    selector = FeatureSelector(corr_threshold=0.98)

    X_train_p = preprocessor.fit_transform(X_train)
    X_train_f = engineer.fit_transform(X_train_p)
    X_train_s = selector.fit_transform(X_train_f)

    X_valid_p = preprocessor.transform(X_valid)
    X_valid_f = engineer.transform(X_valid_p)
    X_valid_s = selector.transform(X_valid_f)

    model = CreditRiskEnsembleModel(config)
    model.fit(X_train_s, y_train)

    valid_prob = model.predict_proba(X_valid_s)
    metrics = classification_metrics(
        y_true=y_valid,
        y_prob=valid_prob,
        threshold=config["training"]["threshold"],
    )

    model_dir = ROOT / config["paths"]["model_dir"]
    report_dir = ROOT / config["paths"]["report_dir"]
    figure_dir = ROOT / config["paths"]["figure_dir"]

    model_path = model_dir / config["files"]["model_file"]
    metrics_path = report_dir / config["files"]["metrics_file"]
    feature_importance_path = figure_dir / "feature_importance.png"

    bundle = {
        "preprocessor": preprocessor,
        "engineer": engineer,
        "selector": selector,
        "model": model,
        "config": config,
    }
    joblib.dump(bundle, model_path)

    save_metrics(metrics, metrics_path)

    fi_df = model.feature_importance()
    save_feature_importance_plot(fi_df, feature_importance_path)

    logger.info(f"训练完成，模型已保存到: {model_path}")
    logger.info(f"评估指标: {metrics}")

    return metrics


if __name__ == "__main__":
    train()