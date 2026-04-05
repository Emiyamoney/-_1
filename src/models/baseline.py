import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class CreditRiskEnsembleModel:
    def __init__(self, config: dict):
        lr_cfg = config["model"]["logistic_regression"]
        rf_cfg = config["model"]["random_forest"]

        self.scaler = StandardScaler()
        self.lr_model = LogisticRegression(
            C=lr_cfg["C"],
            max_iter=lr_cfg["max_iter"],
            class_weight=lr_cfg["class_weight"],
            random_state=config["project"]["random_state"],
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_split=rf_cfg["min_samples_split"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            class_weight=rf_cfg["class_weight"],
            n_jobs=rf_cfg["n_jobs"],
            random_state=config["project"]["random_state"],
        )

        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names_ = X.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)

        self.lr_model.fit(X_scaled, y)
        self.rf_model.fit(X, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names_]

        X_scaled = self.scaler.transform(X)

        lr_prob = self.lr_model.predict_proba(X_scaled)[:, 1]
        rf_prob = self.rf_model.predict_proba(X)[:, 1]

        # 简单平均集成
        final_prob = 0.5 * lr_prob + 0.5 * rf_prob
        return final_prob

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def feature_importance(self) -> pd.DataFrame:
        rf_importance = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": self.rf_model.feature_importances_
        }).sort_values("importance", ascending=False)

        return rf_importance