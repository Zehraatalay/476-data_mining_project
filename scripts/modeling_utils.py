import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# PATH HELPERS
# =========================================================

def get_project_base_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =========================================================
# DATA LOADING
# =========================================================

def load_holdout_model_ready(base_path=None):
    if base_path is None:
        base_path = os.path.join(get_project_base_dir(), "data", "processed", "model_ready", "holdout")

    train_path = os.path.join(base_path, "train_holdout_model_ready.csv")
    val_path = os.path.join(base_path, "validation_holdout_model_ready.csv")
    test_path = os.path.join(base_path, "test_holdout_model_ready.csv")

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    val_df = pd.read_csv(val_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    return train_df, val_df, test_df


def load_kaggle_model_ready(base_path=None):
    if base_path is None:
        base_path = os.path.join(get_project_base_dir(), "data", "processed", "model_ready", "kaggle")

    test_path = os.path.join(base_path, "kaggle_test_model_ready.csv")
    kaggle_test_df = pd.read_csv(test_path, parse_dates=["date"])
    return kaggle_test_df


def load_cv_fold_model_ready(fold: int, base_path=None):
    if base_path is None:
        base_path = os.path.join(get_project_base_dir(), "data", "processed", "model_ready", "timeseries_cv")

    train_path = os.path.join(base_path, f"fold_{fold}_train_model_ready.csv")
    val_path = os.path.join(base_path, f"fold_{fold}_validation_model_ready.csv")

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    val_df = pd.read_csv(val_path, parse_dates=["date"])

    return train_df, val_df


# =========================================================
# FEATURE / TARGET SPLIT
# =========================================================

def get_feature_columns(df: pd.DataFrame, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ["id", "date", "sales"]

    return [col for col in df.columns if col not in exclude_cols]


def split_features_target(df: pd.DataFrame, target_col="sales", exclude_cols=None):
    feature_cols = get_feature_columns(df, exclude_cols=exclude_cols)
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    return X, y, feature_cols


# =========================================================
# METRICS
# =========================================================

def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.maximum(y_pred, 0)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mape(y_true, y_pred)),
        "RMSLE": float(rmsle(y_true, y_pred)),
    }


def print_metrics(metrics: dict, model_name="Model"):
    print(f"\n{model_name}")
    print("-" * len(model_name))
    for k, v in metrics.items():
        if pd.isna(v):
            print(f"{k:<6}: NaN")
        else:
            print(f"{k:<6}: {v:.4f}")


# =========================================================
# OUTPUT HELPERS
# =========================================================

def get_model_output_dir(model_name: str):
    base_dir = os.path.join(get_project_base_dir(), "outputs", "models", model_name.lower().replace(" ", "_"))
    ensure_dir(base_dir)
    return base_dir


def save_metrics_csv(metrics: dict, filepath: str):
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)


def save_predictions_csv(df_with_predictions: pd.DataFrame, filepath: str):
    df_with_predictions.to_csv(filepath, index=False)


def save_text_summary(text: str, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(data: dict, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# KAGGLE SUBMISSION
# =========================================================

def fit_and_predict_kaggle(model, train_full_df: pd.DataFrame, kaggle_test_df: pd.DataFrame):
    X_train, y_train, feature_cols = split_features_target(train_full_df)
    X_test = kaggle_test_df[feature_cols].copy()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    submission = pd.DataFrame({
        "id": kaggle_test_df["id"],
        "sales": preds
    })

    return submission


def save_submission(submission_df: pd.DataFrame, filename="submission.csv", output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(get_project_base_dir(), "outputs", "submissions")

    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, filename)
    submission_df.to_csv(out_path, index=False)
    print(f"\nSubmission kaydedildi: {out_path}")
    return out_path