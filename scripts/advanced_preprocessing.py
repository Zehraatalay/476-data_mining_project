import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class ModelingPreprocessor:
    """
    Akademik olarak temiz preprocessing:
    - train üzerinde fit
    - validation/test üzerinde transform
    """

    def __init__(self):
        self.family_zero_ratio = None
        self.store_zero_ratio = None
        self.numeric_cols_ = None
        self.categorical_cols_ = None
        self.scaler_ = None
        self.encoder_ = None
        self.encoded_feature_names_ = None
        self.feature_columns_ = None

    def _compute_group_sparsity(self, train_df: pd.DataFrame):
        tmp = train_df.copy()
        tmp["is_zero_sales"] = (tmp["sales"] == 0).astype(int)

        self.family_zero_ratio = (
            tmp.groupby("family")["is_zero_sales"]
            .mean()
            .reset_index()
            .rename(columns={"is_zero_sales": "family_zero_ratio"})
        )

        self.store_zero_ratio = (
            tmp.groupby("store_nbr")["is_zero_sales"]
            .mean()
            .reset_index()
            .rename(columns={"is_zero_sales": "store_zero_ratio"})
        )

    def _attach_sparsity_features(self, df: pd.DataFrame):
        out = df.copy()

        if self.family_zero_ratio is not None:
            out = out.merge(self.family_zero_ratio, on="family", how="left")
        if self.store_zero_ratio is not None:
            out = out.merge(self.store_zero_ratio, on="store_nbr", how="left")

        if "family_zero_ratio" in out.columns:
            out["family_zero_ratio"] = out["family_zero_ratio"].fillna(
                self.family_zero_ratio["family_zero_ratio"].mean()
            )
        if "store_zero_ratio" in out.columns:
            out["store_zero_ratio"] = out["store_zero_ratio"].fillna(
                self.store_zero_ratio["store_zero_ratio"].mean()
            )

        return out

    def _select_columns(self, df: pd.DataFrame):
        numeric_candidates = [
            "store_nbr", "cluster", "onpromotion", "promo_log",
            "dcoilwtico", "transactions", "is_holiday", "is_wageday",
            "is_earthquake_period", "year", "month", "day_of_week",
            "day_of_month", "week_of_year", "is_weekend", "quarter",
            "is_month_start", "is_month_end",
            "sin_day_of_week", "cos_day_of_week", "sin_month", "cos_month",
            "family_zero_ratio", "store_zero_ratio"
        ]

        categorical_candidates = [
            "family", "city", "state", "type",
            "holiday_type", "holiday_locale", "holiday_locale_name"
        ]

        numeric_cols = [c for c in numeric_candidates if c in df.columns]
        categorical_cols = [c for c in categorical_candidates if c in df.columns]

        return numeric_cols, categorical_cols

    def fit(self, train_df: pd.DataFrame):
        train_df = train_df.copy()

        if "sales" not in train_df.columns:
            raise ValueError("ModelingPreprocessor.fit için train_df içinde 'sales' olmalı.")

        # 1) Train-based sparsity stats
        self._compute_group_sparsity(train_df)

        # 2) Attach train-based features
        train_df = self._attach_sparsity_features(train_df)

        # 3) Select columns
        self.numeric_cols_, self.categorical_cols_ = self._select_columns(train_df)

        # 4) Numeric scaler
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(train_df[self.numeric_cols_])

        # 5) OneHotEncoder
        try:
            self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse=False)

        self.encoder_.fit(train_df[self.categorical_cols_])
        self.encoded_feature_names_ = self.encoder_.get_feature_names_out(self.categorical_cols_).tolist()

        # 6) Final feature order
        self.feature_columns_ = self.numeric_cols_ + self.encoded_feature_names_

        return self

    def transform(self, df: pd.DataFrame):
        df = df.copy()

        # attach train-based sparsity
        df = self._attach_sparsity_features(df)

        # numeric
        num_scaled = pd.DataFrame(
            self.scaler_.transform(df[self.numeric_cols_]),
            columns=self.numeric_cols_,
            index=df.index
        )

        # categorical
        cat_encoded = pd.DataFrame(
            self.encoder_.transform(df[self.categorical_cols_]),
            columns=self.encoded_feature_names_,
            index=df.index
        )

        X = pd.concat([num_scaled, cat_encoded], axis=1)

        output = pd.concat(
            [
                df[[c for c in ["id", "date", "sales"] if c in df.columns]].copy(),
                X
            ],
            axis=1
        )

        return output

    def fit_transform(self, train_df: pd.DataFrame):
        self.fit(train_df)
        return self.transform(train_df)