import numpy as np
import pandas as pd


class CreditDataPreprocessor:
    def __init__(self):
        self.numeric_fill_values_ = None
        self.feature_columns_ = None

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 年龄异常修正
        if "age" in df.columns:
            df["age"] = df["age"].clip(lower=18, upper=100)

        # 收入负值修正
        if "MonthlyIncome" in df.columns:
            df["MonthlyIncome"] = df["MonthlyIncome"].clip(lower=0)

        # 比率型变量剪裁，防止极端值影响
        for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
            if col in df.columns:
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=0, upper=upper)

        # 逾期次数变量剪裁
        overdue_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
        ]
        for col in overdue_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0, upper=20)

        # 其他计数变量
        count_cols = [
            "NumberOfOpenCreditLinesAndLoans",
            "NumberRealEstateLoansOrLines",
            "NumberOfDependents",
        ]
        for col in count_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        return df

    def fit(self, X: pd.DataFrame):
        X = self._basic_cleaning(X)
        self.feature_columns_ = X.columns.tolist()
        self.numeric_fill_values_ = X.median(numeric_only=True).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self._basic_cleaning(X)

        # 补齐列
        for col in self.feature_columns_:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self.feature_columns_]

        # 中位数填充
        for col, value in self.numeric_fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)

        # 其余缺失值再补0
        X = X.fillna(0)

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)