import pandas as pd


class FeatureSelector:
    def __init__(self, corr_threshold: float = 0.98):
        self.corr_threshold = corr_threshold
        self.selected_columns_ = None

    def fit(self, X: pd.DataFrame):
        X = X.copy()

        # 只保留数值列
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        X_num = X[numeric_cols]

        # 去掉高度相关列
        corr_matrix = X_num.corr().abs()
        upper = corr_matrix.where(
            ~pd.DataFrame(
                [[i >= j for j in range(corr_matrix.shape[1])] for i in range(corr_matrix.shape[0])],
                index=corr_matrix.index,
                columns=corr_matrix.columns,
            )
        )

        drop_cols = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
        self.selected_columns_ = [col for col in numeric_cols if col not in drop_cols]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        return X[self.selected_columns_]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)