import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self):
        self.feature_columns_ = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        eps = 1e-6

        if {"DebtRatio", "MonthlyIncome"}.issubset(df.columns):
            df["DebtPerIncome"] = df["DebtRatio"] / (df["MonthlyIncome"] + eps)

        if {"NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate", "NumberOfTime60-89DaysPastDueNotWorse"}.issubset(df.columns):
            df["TotalPastDueCount"] = (
                df["NumberOfTime30-59DaysPastDueNotWorse"]
                + df["NumberOfTimes90DaysLate"]
                + df["NumberOfTime60-89DaysPastDueNotWorse"]
            )

        if {"NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"}.issubset(df.columns):
            df["LoanLineRatio"] = df["NumberRealEstateLoansOrLines"] / (
                df["NumberOfOpenCreditLinesAndLoans"] + 1
            )

        if {"MonthlyIncome", "NumberOfDependents"}.issubset(df.columns):
            df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

        if "age" in df.columns:
            df["AgeBin"] = pd.cut(
                df["age"],
                bins=[18, 25, 35, 45, 55, 65, 100],
                labels=False,
                include_lowest=True,
            ).astype(float)

        if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
            df["HighUtilizationFlag"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 0.8).astype(int)

        if "DebtRatio" in df.columns:
            df["HighDebtFlag"] = (df["DebtRatio"] > 1.0).astype(int)

        if "MonthlyIncome" in df.columns:
            df["IncomeMissingFlag"] = df["MonthlyIncome"].isna().astype(int)

        return df

    def fit(self, X: pd.DataFrame):
        X_new = self._create_features(X)
        self.feature_columns_ = X_new.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_new = self._create_features(X)

        for col in self.feature_columns_:
            if col not in X_new.columns:
                X_new[col] = 0

        X_new = X_new[self.feature_columns_]
        return X_new

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)