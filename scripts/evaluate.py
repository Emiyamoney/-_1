import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.logger import get_logger


def evaluate():
    logger = get_logger()
    metrics_path = ROOT / "results" / "reports" / "metrics.json"

    if not metrics_path.exists():
        logger.info("还没有 metrics.json，请先运行 train.py")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    logger.info("模型评估结果如下：")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    return metrics


if __name__ == "__main__":
    evaluate()