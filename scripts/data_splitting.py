import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import os


def split_by_date(df: pd.DataFrame,
                  train_end: str = "2016-01-01",
                  val_end: str = "2017-01-01"):
    out = df.sort_values("date").reset_index(drop=True).copy()

    train_df = out[out["date"] < train_end].copy()
    val_df = out[(out["date"] >= train_end) & (out["date"] < val_end)].copy()
    test_df = out[out["date"] >= val_end].copy()

    return train_df, val_df, test_df


def create_holdout_summary(train_df, val_df, test_df, train_end, val_end):
    total = len(train_df) + len(val_df) + len(test_df)

    return pd.DataFrame([
        {
            "split_type": "holdout",
            "subset": "train",
            "rule": f"date < {train_end}",
            "n_rows": len(train_df),
            "ratio": len(train_df) / total,
            "start_date": train_df["date"].min(),
            "end_date": train_df["date"].max()
        },
        {
            "split_type": "holdout",
            "subset": "validation",
            "rule": f"{train_end} <= date < {val_end}",
            "n_rows": len(val_df),
            "ratio": len(val_df) / total,
            "start_date": val_df["date"].min(),
            "end_date": val_df["date"].max()
        },
        {
            "split_type": "holdout",
            "subset": "test",
            "rule": f"date >= {val_end}",
            "n_rows": len(test_df),
            "ratio": len(test_df) / total,
            "start_date": test_df["date"].min(),
            "end_date": test_df["date"].max()
        }
    ])


def print_holdout_summary(summary_df):
    print("\n--- HOLDOUT SPLIT SUMMARY ---")
    print(summary_df.to_string(index=False))


def get_time_series_cv_splits(df: pd.DataFrame, n_splits: int = 5):
    out = df.sort_values("date").reset_index(drop=True).copy()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    splits = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(out), start=1):
        fold_train = out.iloc[train_idx].copy()
        fold_val = out.iloc[val_idx].copy()

        splits.append({
            "fold": fold,
            "train_df": fold_train,
            "val_df": fold_val
        })

    return splits


def create_time_series_cv_summary(df: pd.DataFrame, n_splits: int = 5):
    splits = get_time_series_cv_splits(df, n_splits=n_splits)

    rows = []
    for split in splits:
        fold = split["fold"]
        fold_train = split["train_df"]
        fold_val = split["val_df"]

        rows.append({
            "split_type": "timeseries_cv",
            "fold": fold,
            "train_size": len(fold_train),
            "validation_size": len(fold_val),
            "train_start": fold_train["date"].min(),
            "train_end": fold_train["date"].max(),
            "validation_start": fold_val["date"].min(),
            "validation_end": fold_val["date"].max()
        })

    return pd.DataFrame(rows)


def print_time_series_cv_summary(cv_summary_df):
    print("\n--- TIME SERIES CV SUMMARY ---")
    print(cv_summary_df.to_string(index=False))


def save_split_outputs(holdout_summary_df, cv_summary_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    holdout_summary_df.to_csv(os.path.join(output_dir, "holdout_split_summary.csv"), index=False)
    cv_summary_df.to_csv(os.path.join(output_dir, "timeseries_cv_summary.csv"), index=False)


def save_time_series_cv_folds(df: pd.DataFrame, output_dir: str, n_splits: int = 5):
    os.makedirs(output_dir, exist_ok=True)

    splits = get_time_series_cv_splits(df, n_splits=n_splits)

    for split in splits:
        fold = split["fold"]
        split["train_df"].to_csv(os.path.join(output_dir, f"fold_{fold}_train.csv"), index=False)
        split["val_df"].to_csv(os.path.join(output_dir, f"fold_{fold}_validation.csv"), index=False)

    print(f"\nTime-series CV fold dosyaları kaydedildi: {output_dir}")