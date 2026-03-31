import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from modeling_utils import (
    regression_metrics,
    print_metrics,
    get_model_output_dir,
    save_metrics_csv,
    save_predictions_csv,
    save_text_summary,
    save_json,
    get_project_base_dir,
)
from sampling_utils import (
    sample_ids_by_store_family,
    print_sampling_summary
)


MODEL_NAME = "LSTM"


# =========================================================
# HELPERS
# =========================================================

def log_step(message: str):
    print(f"\n[INFO] {message}", flush=True)


def timed_block_start():
    return time.time()


def timed_block_end(start_time, label: str):
    elapsed = time.time() - start_time
    print(f"[DONE] {label} | elapsed: {elapsed:.2f} sec", flush=True)


def load_holdout_raw_ready():
    base_dir = get_project_base_dir()
    base_path = os.path.join(base_dir, "data", "processed", "holdout")

    train_df = pd.read_csv(
        os.path.join(base_path, "train_holdout_ready.csv"),
        parse_dates=["date"],
        low_memory=False
    )
    val_df = pd.read_csv(
        os.path.join(base_path, "validation_holdout_ready.csv"),
        parse_dates=["date"],
        low_memory=False
    )
    test_df = pd.read_csv(
        os.path.join(base_path, "test_holdout_ready.csv"),
        parse_dates=["date"],
        low_memory=False
    )

    return train_df, val_df, test_df


def get_lstm_output_dir():
    output_dir = get_model_output_dir(MODEL_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =========================================================
# FEATURE SET
# =========================================================

def get_lstm_feature_columns(df: pd.DataFrame):
    candidate_cols = [
        "onpromotion",
        "promo_log",
        "dcoilwtico",
        "transactions",
        "is_holiday",
        "is_wageday",
        "is_earthquake_period",
        "year",
        "month",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "is_weekend",
        "quarter",
        "is_month_start",
        "is_month_end",
        "sin_day_of_week",
        "cos_day_of_week",
        "sin_month",
        "cos_month",
    ]
    return [c for c in candidate_cols if c in df.columns]


# =========================================================
# PREP
# =========================================================

def attach_sparsity_features(train_df: pd.DataFrame, other_df: pd.DataFrame):
    """
    family/store zero ratio train'den öğrenilir, other_df'ye uygulanır.
    """
    train_df = train_df.copy()
    other_df = other_df.copy()

    train_df["is_zero_sales"] = (train_df["sales"] == 0).astype(int)

    family_zero_ratio = (
        train_df.groupby("family")["is_zero_sales"]
        .mean()
        .reset_index()
        .rename(columns={"is_zero_sales": "family_zero_ratio"})
    )

    store_zero_ratio = (
        train_df.groupby("store_nbr")["is_zero_sales"]
        .mean()
        .reset_index()
        .rename(columns={"is_zero_sales": "store_zero_ratio"})
    )

    train_df = train_df.merge(family_zero_ratio, on="family", how="left")
    train_df = train_df.merge(store_zero_ratio, on="store_nbr", how="left")

    other_df = other_df.merge(family_zero_ratio, on="family", how="left")
    other_df = other_df.merge(store_zero_ratio, on="store_nbr", how="left")

    other_df["family_zero_ratio"] = other_df["family_zero_ratio"].fillna(
        family_zero_ratio["family_zero_ratio"].mean()
    )
    other_df["store_zero_ratio"] = other_df["store_zero_ratio"].fillna(
        store_zero_ratio["store_zero_ratio"].mean()
    )

    return train_df, other_df


def sample_holdout_entities(train_df, val_df, test_df, sample_ratio=0.20, random_state=42):
    sampled_train_ids, sampled_entities, sampled_train_raw = sample_ids_by_store_family(
        train_df,
        sample_ratio=sample_ratio,
        random_state=random_state
    )
    print_sampling_summary(train_df, sampled_train_raw, sampled_entities)

    sampled_val = val_df.merge(sampled_entities, on=["store_nbr", "family"], how="inner").copy()
    sampled_test = test_df.merge(sampled_entities, on=["store_nbr", "family"], how="inner").copy()

    sampled_train = train_df[train_df["id"].isin(sampled_train_ids)].copy()

    sampled_train = sampled_train.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    sampled_val = sampled_val.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    sampled_test = sampled_test.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    return sampled_train, sampled_val, sampled_test, sampled_entities


# =========================================================
# SEQUENCE BUILDING
# =========================================================

def create_sequences_from_df(df, feature_cols, window_size=14):
    """
    Her store_nbr + family sequence'i için sliding window üretir.
    Target = t anındaki sales
    Features = t-window_size ... t-1 aralığındaki feature vektörleri
    """
    X_list = []
    y_list = []
    meta_rows = []

    df = df.sort_values(["store_nbr", "family", "date"]).copy()

    for (store_nbr, family), group in df.groupby(["store_nbr", "family"]):
        group = group.sort_values("date").reset_index(drop=True)

        if len(group) <= window_size:
            continue

        feat_values = group[feature_cols].values.astype(np.float32)
        target_values = group["sales"].values.astype(np.float32)

        for i in range(window_size, len(group)):
            X_seq = feat_values[i - window_size:i]
            y = target_values[i]

            X_list.append(X_seq)
            y_list.append(y)

            meta_rows.append({
                "id": group.loc[i, "id"],
                "date": group.loc[i, "date"],
                "store_nbr": store_nbr,
                "family": family,
                "sales": y
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df


def scale_sequence_features(X_train, X_other):
    """
    3D tensor için scaler:
    scaler sadece train üzerinde fit edilir.
    """
    n_train, t_train, f_train = X_train.shape
    n_other, t_other, f_other = X_other.shape

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, f_train)
    X_other_2d = X_other.reshape(-1, f_other)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, t_train, f_train)
    X_other_scaled = scaler.transform(X_other_2d).reshape(n_other, t_other, f_other)

    return X_train_scaled, X_other_scaled, scaler


# =========================================================
# MODEL
# =========================================================

def build_lstm_model(input_shape, learning_rate=0.001):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


# =========================================================
# HOLDOUT EVALUATION
# =========================================================

def evaluate_holdout_lstm(
    sample_ratio=0.20,
    random_state=42,
    window_size=14,
    epochs=20,
    batch_size=256,
    learning_rate=0.001
):
    output_dir = get_lstm_output_dir()

    log_step("LSTM holdout evaluation başlıyor")
    train_raw, val_raw, test_raw = load_holdout_raw_ready()

    # -------------------------------------------------
    # 1) Sampling
    # -------------------------------------------------
    log_step(f"Sampling başlıyor | ratio={sample_ratio}")
    sampled_train, sampled_val, sampled_test, sampled_entities = sample_holdout_entities(
        train_raw, val_raw, test_raw,
        sample_ratio=sample_ratio,
        random_state=random_state
    )
    sampled_entities.to_csv(os.path.join(output_dir, "sampled_entities.csv"), index=False)

    # -------------------------------------------------
    # 2) VALIDATION STAGE
    # -------------------------------------------------
    sampled_train_valstage, sampled_val_valstage = attach_sparsity_features(
        sampled_train.copy(),
        sampled_val.copy()
    )

    base_feature_cols = get_lstm_feature_columns(sampled_train_valstage)
    extra_cols = ["family_zero_ratio", "store_zero_ratio"]
    feature_cols = base_feature_cols + [c for c in extra_cols if c in sampled_train_valstage.columns]

    log_step("Validation sequence'leri oluşturuluyor")
    X_train, y_train, train_meta = create_sequences_from_df(
        sampled_train_valstage, feature_cols, window_size=window_size
    )
    X_val, y_val, val_meta = create_sequences_from_df(
        sampled_val_valstage, feature_cols, window_size=window_size
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train/Validation sequence üretilemedi. window_size çok büyük olabilir.")

    X_train_scaled, X_val_scaled, scaler = scale_sequence_features(X_train, X_val)

    log_step("LSTM validation training başlıyor")
    start = timed_block_start()

    model_val = build_lstm_model(
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        learning_rate=learning_rate
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history_val = model_val.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    val_preds = model_val.predict(X_val_scaled, verbose=0).flatten()
    timed_block_end(start, "Validation training tamamlandı")

    val_metrics = regression_metrics(y_val, val_preds)
    print_metrics(val_metrics, model_name="LSTM Validation Results")

    val_pred_df = val_meta.copy()
    val_pred_df["prediction"] = val_preds

    save_metrics_csv(val_metrics, os.path.join(output_dir, "validation_metrics.csv"))
    save_predictions_csv(val_pred_df, os.path.join(output_dir, "validation_predictions.csv"))
    pd.DataFrame(history_val.history).to_csv(
        os.path.join(output_dir, "validation_training_history.csv"),
        index=False
    )

    # -------------------------------------------------
    # 3) TEST STAGE
    # -------------------------------------------------
    sampled_train_plus_val_raw = pd.concat([sampled_train, sampled_val], axis=0).reset_index(drop=True)
    sampled_train_plus_val_teststage, sampled_test_teststage = attach_sparsity_features(
        sampled_train_plus_val_raw.copy(),
        sampled_test.copy()
    )

    base_feature_cols_test = get_lstm_feature_columns(sampled_train_plus_val_teststage)
    feature_cols_test = base_feature_cols_test + [
        c for c in ["family_zero_ratio", "store_zero_ratio"]
        if c in sampled_train_plus_val_teststage.columns
    ]

    log_step("Test sequence'leri oluşturuluyor")
    X_train_full, y_train_full, train_full_meta = create_sequences_from_df(
        sampled_train_plus_val_teststage, feature_cols_test, window_size=window_size
    )
    X_test, y_test, test_meta = create_sequences_from_df(
        sampled_test_teststage, feature_cols_test, window_size=window_size
    )

    if len(X_test) == 0:
        raise ValueError("Test sequence üretilemedi. window_size çok büyük olabilir.")

    X_train_full_scaled, X_test_scaled, scaler_test = scale_sequence_features(X_train_full, X_test)

    log_step("LSTM final test training başlıyor")
    start = timed_block_start()

    model_test = build_lstm_model(
        input_shape=(X_train_full_scaled.shape[1], X_train_full_scaled.shape[2]),
        learning_rate=learning_rate
    )

    early_stopping_test = EarlyStopping(
        monitor="loss",
        patience=2,
        restore_best_weights=True
    )

    history_test = model_test.fit(
        X_train_full_scaled,
        y_train_full,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping_test],
        verbose=1
    )

    test_preds = model_test.predict(X_test_scaled, verbose=0).flatten()
    timed_block_end(start, "Test training tamamlandı")

    test_metrics = regression_metrics(y_test, test_preds)
    print_metrics(test_metrics, model_name="LSTM Test Results")

    test_pred_df = test_meta.copy()
    test_pred_df["prediction"] = test_preds

    save_metrics_csv(test_metrics, os.path.join(output_dir, "test_metrics.csv"))
    save_predictions_csv(test_pred_df, os.path.join(output_dir, "test_predictions.csv"))
    pd.DataFrame(history_test.history).to_csv(
        os.path.join(output_dir, "test_training_history.csv"),
        index=False
    )

    summary_text = f"""
Model: LSTM

Sampling:
- Method: store-family based representative sampling
- Sample ratio: {sample_ratio}

Sequence Parameters:
- window_size = {window_size}

Training Parameters:
- epochs = {epochs}
- batch_size = {batch_size}
- learning_rate = {learning_rate}

Validation Strategy:
- Train set used for fitting
- Validation set used for monitoring / model selection

Final Test Strategy:
- Train + Validation merged for final fitting
- Test set used for final evaluation

Validation Feature Columns:
{feature_cols}

Test Feature Columns:
{feature_cols_test}

Validation Metrics:
{val_metrics}

Test Metrics:
{test_metrics}
""".strip()

    save_text_summary(summary_text, os.path.join(output_dir, "holdout_summary.txt"))
    save_json(
        {
            "model": MODEL_NAME,
            "sample_ratio": sample_ratio,
            "window_size": window_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_feature_columns": feature_cols,
            "test_feature_columns": feature_cols_test,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics
        },
        os.path.join(output_dir, "holdout_summary.json")
    )

    return {
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics
    }


def main():
    print("\n" + "=" * 80)
    print("LSTM MODEL TRAINING")
    print("=" * 80, flush=True)

    evaluate_holdout_lstm(
        sample_ratio=0.20,
        random_state=42,
        window_size=14,
        epochs=20,
        batch_size=256,
        learning_rate=0.001
    )

    print("\n" + "=" * 80)
    print("LSTM TAMAMLANDI")
    print("=" * 80)
    print("Kaydedilen çıktı klasörü:")
    print(get_lstm_output_dir(), flush=True)


if __name__ == "__main__":
    main()