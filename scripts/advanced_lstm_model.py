import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Concatenate,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

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


MODEL_NAME = "LSTM_Advanced"


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


def get_output_dir():
    output_dir = get_model_output_dir(MODEL_NAME)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def get_sequence_feature_columns(df: pd.DataFrame):
    """
    Sequence içine girecek numeric feature'lar.
    DİKKAT: geçmiş sales'i de ekliyoruz.
    """
    candidate_cols = [
        "sales",               # geçmiş target
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
        "family_zero_ratio",
        "store_zero_ratio",
    ]
    return [c for c in candidate_cols if c in df.columns]


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
# LABEL ENCODING FOR EMBEDDINGS
# =========================================================

def fit_entity_encoders(train_df: pd.DataFrame):
    store_values = sorted(train_df["store_nbr"].dropna().unique().tolist())
    family_values = sorted(train_df["family"].dropna().unique().tolist())

    store_to_idx = {v: i + 1 for i, v in enumerate(store_values)}   # 0 reserved unknown
    family_to_idx = {v: i + 1 for i, v in enumerate(family_values)}

    return store_to_idx, family_to_idx


def apply_entity_encoders(df: pd.DataFrame, store_to_idx: dict, family_to_idx: dict):
    out = df.copy()
    out["store_idx"] = out["store_nbr"].map(store_to_idx).fillna(0).astype(int)
    out["family_idx"] = out["family"].map(family_to_idx).fillna(0).astype(int)
    return out


# =========================================================
# SEQUENCE BUILDING
# =========================================================

def create_sequences_with_entities(df, feature_cols, window_size=28, target_log=True):
    """
    Her store-family için:
    - X_seq: geçmiş window_size günün numeric feature sequence'i
    - store_idx / family_idx: target satırındaki entity bilgisi
    - y: t anındaki sales (opsiyonel log1p)
    """
    X_list = []
    y_list = []
    store_idx_list = []
    family_idx_list = []
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

            y_raw = target_values[i]
            y = np.log1p(y_raw) if target_log else y_raw

            X_list.append(X_seq)
            y_list.append(y)
            store_idx_list.append(int(group.loc[i, "store_idx"]))
            family_idx_list.append(int(group.loc[i, "family_idx"]))

            meta_rows.append({
                "id": group.loc[i, "id"],
                "date": group.loc[i, "date"],
                "store_nbr": store_nbr,
                "family": family,
                "sales": y_raw
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    store_idx = np.array(store_idx_list, dtype=np.int32)
    family_idx = np.array(family_idx_list, dtype=np.int32)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, store_idx, family_idx, meta_df


def scale_sequence_features(X_train, X_other):
    """
    Sequence numeric feature'lar train üzerinden ölçeklenir.
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

def build_advanced_lstm_model(
    seq_input_shape,
    n_store_tokens,
    n_family_tokens,
    learning_rate=0.001
):
    seq_input = Input(shape=seq_input_shape, name="seq_input")
    store_input = Input(shape=(1,), name="store_input")
    family_input = Input(shape=(1,), name="family_input")

    # sequence branch
    x_seq = LSTM(128, return_sequences=True)(seq_input)
    x_seq = Dropout(0.2)(x_seq)
    x_seq = LSTM(64, return_sequences=False)(x_seq)
    x_seq = Dropout(0.2)(x_seq)
    x_seq = Dense(64, activation="relu")(x_seq)
    x_seq = BatchNormalization()(x_seq)

    # store embedding
    store_emb_dim = min(16, max(4, n_store_tokens // 4))
    x_store = Embedding(input_dim=n_store_tokens + 1, output_dim=store_emb_dim, name="store_embedding")(store_input)
    x_store = Flatten()(x_store)

    # family embedding
    family_emb_dim = min(16, max(4, n_family_tokens // 4))
    x_family = Embedding(input_dim=n_family_tokens + 1, output_dim=family_emb_dim, name="family_embedding")(family_input)
    x_family = Flatten()(x_family)

    # combine
    x = Concatenate()([x_seq, x_store, x_family])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, name="sales_output")(x)

    model = Model(
        inputs=[seq_input, store_input, family_input],
        outputs=output
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=Huber(delta=1.0)
    )
    return model


# =========================================================
# HOLDOUT EVALUATION
# =========================================================

def evaluate_holdout_lstm_advanced(
    sample_ratio=0.20,
    random_state=42,
    window_size=28,
    epochs=30,
    batch_size=256,
    learning_rate=0.001
):
    output_dir = get_output_dir()

    log_step("Advanced LSTM holdout evaluation başlıyor")
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

    store_to_idx, family_to_idx = fit_entity_encoders(sampled_train_valstage)
    sampled_train_valstage = apply_entity_encoders(sampled_train_valstage, store_to_idx, family_to_idx)
    sampled_val_valstage = apply_entity_encoders(sampled_val_valstage, store_to_idx, family_to_idx)

    feature_cols = get_sequence_feature_columns(sampled_train_valstage)

    log_step("Validation sequence'leri oluşturuluyor")
    X_train, y_train, store_train, family_train, train_meta = create_sequences_with_entities(
        sampled_train_valstage, feature_cols, window_size=window_size, target_log=True
    )
    X_val, y_val, store_val, family_val, val_meta = create_sequences_with_entities(
        sampled_val_valstage, feature_cols, window_size=window_size, target_log=True
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train/Validation sequence üretilemedi. window_size çok büyük olabilir.")

    X_train_scaled, X_val_scaled, scaler = scale_sequence_features(X_train, X_val)

    log_step("Advanced LSTM validation training başlıyor")
    start = timed_block_start()

    model_path = os.path.join(output_dir, "best_validation_model.keras")

    model_val = build_advanced_lstm_model(
        seq_input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        n_store_tokens=max(store_to_idx.values()) if len(store_to_idx) > 0 else 1,
        n_family_tokens=max(family_to_idx.values()) if len(family_to_idx) > 0 else 1,
        learning_rate=learning_rate
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history_val = model_val.fit(
        [X_train_scaled, store_train, family_train],
        y_train,
        validation_data=([X_val_scaled, store_val, family_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    val_preds_log = model_val.predict([X_val_scaled, store_val, family_val], verbose=0).flatten()
    val_preds = np.expm1(val_preds_log)
    val_preds = np.maximum(val_preds, 0)

    timed_block_end(start, "Validation training tamamlandı")

    val_metrics = regression_metrics(val_meta["sales"].values, val_preds)
    print_metrics(val_metrics, model_name="Advanced LSTM Validation Results")

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

    store_to_idx_test, family_to_idx_test = fit_entity_encoders(sampled_train_plus_val_teststage)
    sampled_train_plus_val_teststage = apply_entity_encoders(
        sampled_train_plus_val_teststage, store_to_idx_test, family_to_idx_test
    )
    sampled_test_teststage = apply_entity_encoders(
        sampled_test_teststage, store_to_idx_test, family_to_idx_test
    )

    feature_cols_test = get_sequence_feature_columns(sampled_train_plus_val_teststage)

    log_step("Test sequence'leri oluşturuluyor")
    X_train_full, y_train_full, store_train_full, family_train_full, train_full_meta = create_sequences_with_entities(
        sampled_train_plus_val_teststage, feature_cols_test, window_size=window_size, target_log=True
    )
    X_test, y_test, store_test, family_test, test_meta = create_sequences_with_entities(
        sampled_test_teststage, feature_cols_test, window_size=window_size, target_log=True
    )

    if len(X_test) == 0:
        raise ValueError("Test sequence üretilemedi. window_size çok büyük olabilir.")

    X_train_full_scaled, X_test_scaled, scaler_test = scale_sequence_features(X_train_full, X_test)

    log_step("Advanced LSTM final model training başlıyor")
    start = timed_block_start()

    model_test_path = os.path.join(output_dir, "best_test_stage_model.keras")

    model_test = build_advanced_lstm_model(
        seq_input_shape=(X_train_full_scaled.shape[1], X_train_full_scaled.shape[2]),
        n_store_tokens=max(store_to_idx_test.values()) if len(store_to_idx_test) > 0 else 1,
        n_family_tokens=max(family_to_idx_test.values()) if len(family_to_idx_test) > 0 else 1,
        learning_rate=learning_rate
    )

    callbacks_test = [
        EarlyStopping(
            monitor="loss",
            patience=4,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5
        ),
        ModelCheckpoint(
            filepath=model_test_path,
            monitor="loss",
            save_best_only=True
        )
    ]

    history_test = model_test.fit(
        [X_train_full_scaled, store_train_full, family_train_full],
        y_train_full,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_test,
        verbose=1
    )

    test_preds_log = model_test.predict([X_test_scaled, store_test, family_test], verbose=0).flatten()
    test_preds = np.expm1(test_preds_log)
    test_preds = np.maximum(test_preds, 0)

    timed_block_end(start, "Final model training + test prediction tamamlandı")

    test_metrics = regression_metrics(test_meta["sales"].values, test_preds)
    print_metrics(test_metrics, model_name="Advanced LSTM Test Results")

    test_pred_df = test_meta.copy()
    test_pred_df["prediction"] = test_preds

    save_metrics_csv(test_metrics, os.path.join(output_dir, "test_metrics.csv"))
    save_predictions_csv(test_pred_df, os.path.join(output_dir, "test_predictions.csv"))
    pd.DataFrame(history_test.history).to_csv(
        os.path.join(output_dir, "test_training_history.csv"),
        index=False
    )

    summary_text = f"""
Model: Advanced LSTM

Sampling:
- Method: store-family based representative sampling
- Sample ratio: {sample_ratio}

Sequence Parameters:
- window_size = {window_size}
- target_transform = log1p(sales)

Training Parameters:
- epochs = {epochs}
- batch_size = {batch_size}
- learning_rate = {learning_rate}
- loss = Huber
- callbacks = EarlyStopping + ReduceLROnPlateau + ModelCheckpoint

Architecture:
- numeric sequence input
- store embedding
- family embedding
- stacked LSTM

Validation Metrics:
{val_metrics}

Test Metrics:
{test_metrics}

Validation Feature Columns:
{feature_cols}

Test Feature Columns:
{feature_cols_test}
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
    print("ADVANCED LSTM MODEL TRAINING")
    print("=" * 80, flush=True)

    evaluate_holdout_lstm_advanced(
        sample_ratio=0.20,
        random_state=42,
        window_size=28,
        epochs=30,
        batch_size=256,
        learning_rate=0.001
    )

    print("\n" + "=" * 80)
    print("ADVANCED LSTM TAMAMLANDI")
    print("=" * 80)
    print("Kaydedilen çıktı klasörü:")
    print(get_output_dir(), flush=True)


if __name__ == "__main__":
    main()