from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_feature_importance_plot(feature_importance_df: pd.DataFrame, output_path: Path, top_n: int = 15):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = feature_importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()