from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_distribution_plot(series: pd.Series, output_path: Path, title: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    series.hist(bins=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()