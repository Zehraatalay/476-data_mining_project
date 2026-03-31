import numpy as np
import pandas as pd


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Split öncesi güvenle uygulanabilecek preprocessing:
    - missing value handling
    - deterministic feature engineering
    Scaling / encoding burada yapılmaz.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # -------------------------------------------------
    # 1) Missing Value Handling
    # -------------------------------------------------

    # Oil
    train_df["dcoilwtico"] = train_df["dcoilwtico"].ffill().bfill()
    test_df["dcoilwtico"] = test_df["dcoilwtico"].ffill().bfill()

    # Transactions
    if "transactions" in train_df.columns:
        train_df["transactions"] = train_df["transactions"].fillna(0)
    if "transactions" in test_df.columns:
        test_df["transactions"] = test_df["transactions"].fillna(0)

    # Holiday text columns
    holiday_text_cols = ["holiday_type", "holiday_locale", "holiday_locale_name", "holiday_description"]
    for col in holiday_text_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("None")
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("None")

    # -------------------------------------------------
    # 2) Deterministic Feature Engineering
    # -------------------------------------------------

    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Wage day
        out["is_wageday"] = out["date"].apply(
            lambda x: 1 if (x.day == 15 or x.is_month_end) else 0
        )

        # Earthquake period
        out["is_earthquake_period"] = (
            (out["date"] >= "2016-04-16") & (out["date"] <= "2016-05-31")
        ).astype(int)

        # Calendar features
        out["year"] = out["date"].dt.year
        out["month"] = out["date"].dt.month
        out["day_of_week"] = out["date"].dt.dayofweek
        out["day_of_month"] = out["date"].dt.day
        out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
        out["quarter"] = out["date"].dt.quarter
        out["is_month_start"] = out["date"].dt.is_month_start.astype(int)
        out["is_month_end"] = out["date"].dt.is_month_end.astype(int)

        # Cyclical encodings
        out["sin_day_of_week"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
        out["cos_day_of_week"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
        out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12)
        out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12)

        # Promotion transform
        if "onpromotion" in out.columns:
            out["promo_log"] = np.log1p(out["onpromotion"])

        return out

    train_df = create_features(train_df)
    test_df = create_features(test_df)

    return train_df, test_df