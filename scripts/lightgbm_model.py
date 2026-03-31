import os
import time
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor

from modeling_utils import (
    load_holdout_model_ready,
    load_cv_fold_model_ready,
    split_features_target,
    regression_metrics,
    print_metrics,
    load_kaggle_model_ready,
    save_submission,
    get_model_output_dir,
    save_metrics_csv,
    save_predictions_csv,
    save_text_summary,
    save_json,
    get_project_base_dir
)
from sampling_utils import (
    sample_ids_by_store_family,
    filter_model_ready_by_ids,
    print_sampling_summary
)


MODEL_NAME = "LightGBM"


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

    train_df = pd.read_csv(os.path.join(base_path, "train_holdout_ready.csv"), parse_dates=["date"])
    val_df = pd.read_csv(os.path.join(base_path, "validation_holdout_ready.csv"), parse_dates=["date"])
    test_df = pd.read_csv(os.path.join(base_path, "test_holdout_ready.csv"), parse_dates=["date"])

    return train_df, val_df, test_df


def load_cv_fold_raw_ready(fold: int):
    base_dir = get_project_base_dir()
    base_path = os.path.join(base_dir, "data", "processed", "timeseries_cv")

    train_df = pd.read_csv(os.path.join(base_path, f"fold_{fold}_train.csv"), parse_dates=["date"])
    val_df = pd.read_csv(os.path.join(base_path, f"fold_{fold}_validation.csv"), parse_dates=["date"])

    return train_df, val_df


# =========================================================
# FEATURE SELECTION
# =========================================================

def select_top_k_features(X_train, y_train, feature_cols, top_k=40, random_state=42):
    log_step(f"Feature selection başlıyor (top_k={top_k})")
    start = timed_block_start()

    selector = ExtraTreesRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    selector.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": selector.feature_importances_
    }).sort_values("importance", ascending=False)

    selected_features = importance_df.head(top_k)["feature"].tolist()

    timed_block_end(start, f"Feature selection tamamlandı (top_k={top_k})")
    return selected_features, importance_df


# =========================================================
# MODEL FACTORY
# =========================================================

def build_lightgbm_model(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
):
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )


# =========================================================
# HOLDOUT EVALUATION
# =========================================================

def evaluate_holdout_lightgbm_single(
    top_k_features,
    sample_ratio=0.20,
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8
):
    run_name = f"top_{top_k_features}"
    output_dir = os.path.join(get_model_output_dir(MODEL_NAME), run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_step(f"HOLDOUT evaluation başlıyor | {run_name}")

    # Raw ve model_ready verileri ayrı yükle
    train_raw, val_raw, test_raw = load_holdout_raw_ready()
    train_model, val_model, test_model = load_holdout_model_ready()

    # -------------------------------------------------
    # 1) SAMPLE TRAIN SET ONLY (raw üzerinden)
    # -------------------------------------------------
    log_step(f"Train holdout örnekleniyor | ratio={sample_ratio}")
    sampled_train_ids, sampled_entities, sampled_train_raw = sample_ids_by_store_family(
        train_raw,
        sample_ratio=sample_ratio,
        random_state=random_state
    )
    print_sampling_summary(train_raw, sampled_train_raw, sampled_entities)

    sampled_entities.to_csv(
        os.path.join(output_dir, "sampled_entities_train.csv"),
        index=False
    )

    # validation/test için aynı entity set
    sampled_val_raw = val_raw.merge(sampled_entities, on=["store_nbr", "family"], how="inner")
    sampled_test_raw = test_raw.merge(sampled_entities, on=["store_nbr", "family"], how="inner")

    sampled_val_ids = sampled_val_raw["id"].tolist()
    sampled_test_ids = sampled_test_raw["id"].tolist()

    sampled_train_model = filter_model_ready_by_ids(train_model, sampled_train_ids)
    sampled_val_model = filter_model_ready_by_ids(val_model, sampled_val_ids)
    sampled_test_model = filter_model_ready_by_ids(test_model, sampled_test_ids)

    # -------------------------------------------------
    # 2) TRAIN -> VALIDATION
    # -------------------------------------------------
    X_train, y_train, feature_cols = split_features_target(sampled_train_model)
    X_val, y_val, _ = split_features_target(sampled_val_model)

    selected_features, importance_df = select_top_k_features(
        X_train, y_train, feature_cols,
        top_k=top_k_features,
        random_state=random_state
    )

    importance_df.to_csv(os.path.join(output_dir, "feature_importance_full.csv"), index=False)
    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        os.path.join(output_dir, "selected_features_validation_stage.csv"),
        index=False
    )

    print(f"[INFO] Selected features ({run_name}): {len(selected_features)}", flush=True)
    print(f"[INFO] First 10 selected features: {selected_features[:10]}", flush=True)

    log_step(f"LightGBM validation training başlıyor | {run_name}")
    start = timed_block_start()

    model_val = build_lightgbm_model(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state
    )

    model_val.fit(X_train[selected_features], y_train)
    val_preds = model_val.predict(X_val[selected_features])

    timed_block_end(start, f"Validation training tamamlandı | {run_name}")

    val_metrics = regression_metrics(y_val, val_preds)
    print_metrics(val_metrics, model_name=f"LightGBM Validation Results ({run_name})")

    val_pred_df = sampled_val_model[["id", "date", "sales"]].copy()
    val_pred_df["prediction"] = val_preds

    save_metrics_csv(val_metrics, os.path.join(output_dir, "validation_metrics.csv"))
    save_predictions_csv(val_pred_df, os.path.join(output_dir, "validation_predictions.csv"))

    # -------------------------------------------------
    # 3) TRAIN + VALIDATION -> TEST
    # -------------------------------------------------
    sampled_train_plus_val_model = pd.concat([sampled_train_model, sampled_val_model], axis=0).reset_index(drop=True)

    X_train_full, y_train_full, feature_cols_full = split_features_target(sampled_train_plus_val_model)
    X_test, y_test, _ = split_features_target(sampled_test_model)

    selected_features_test, importance_df_test = select_top_k_features(
        X_train_full, y_train_full, feature_cols_full,
        top_k=top_k_features,
        random_state=random_state
    )

    importance_df_test.to_csv(os.path.join(output_dir, "feature_importance_train_val.csv"), index=False)
    pd.DataFrame({"selected_feature": selected_features_test}).to_csv(
        os.path.join(output_dir, "selected_features_test_stage.csv"),
        index=False
    )

    log_step(f"LightGBM final test training başlıyor | {run_name}")
    start = timed_block_start()

    model_test = build_lightgbm_model(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state
    )

    model_test.fit(X_train_full[selected_features_test], y_train_full)
    test_preds = model_test.predict(X_test[selected_features_test])

    timed_block_end(start, f"Test training tamamlandı | {run_name}")

    test_metrics = regression_metrics(y_test, test_preds)
    print_metrics(test_metrics, model_name=f"LightGBM Test Results ({run_name})")

    test_pred_df = sampled_test_model[["id", "date", "sales"]].copy()
    test_pred_df["prediction"] = test_preds

    save_metrics_csv(test_metrics, os.path.join(output_dir, "test_metrics.csv"))
    save_predictions_csv(test_pred_df, os.path.join(output_dir, "test_predictions.csv"))

    summary_text = f"""
Model: LightGBM
Run: {run_name}

Sampling:
- Method: store-family based representative sampling
- Sample ratio: {sample_ratio}

Feature Selection:
- Method: ExtraTrees importance ranking
- Top-K Features: {top_k_features}

Parameters:
- n_estimators = {n_estimators}
- learning_rate = {learning_rate}
- max_depth = {max_depth}
- num_leaves = {num_leaves}
- subsample = {subsample}
- colsample_bytree = {colsample_bytree}
- random_state = {random_state}

Validation Metrics:
{val_metrics}

Test Metrics:
{test_metrics}
""".strip()

    save_text_summary(summary_text, os.path.join(output_dir, "holdout_summary.txt"))
    save_json(
        {
            "model": MODEL_NAME,
            "run_name": run_name,
            "sample_ratio": sample_ratio,
            "top_k_features": top_k_features,
            "params": {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "num_leaves": num_leaves,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "random_state": random_state
            },
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics
        },
        os.path.join(output_dir, "holdout_summary.json")
    )

    return {
        "run_name": run_name,
        "sample_ratio": sample_ratio,
        "top_k_features": top_k_features,
        "validation_MAE": val_metrics["MAE"],
        "validation_RMSE": val_metrics["RMSE"],
        "validation_MAPE": val_metrics["MAPE"],
        "validation_RMSLE": val_metrics["RMSLE"],
        "test_MAE": test_metrics["MAE"],
        "test_RMSE": test_metrics["RMSE"],
        "test_MAPE": test_metrics["MAPE"],
        "test_RMSLE": test_metrics["RMSLE"],
    }


# =========================================================
# TIME-SERIES CV
# =========================================================

def evaluate_timeseries_cv_lightgbm_single(
    top_k_features,
    sample_ratio=0.20,
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    n_folds=3
):
    run_name = f"top_{top_k_features}"
    output_dir = os.path.join(get_model_output_dir(MODEL_NAME), run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_step(f"TIME-SERIES CV başlıyor | {run_name}")

    rows = []
    selected_features_summary = []

    for fold in range(1, n_folds + 1):
        log_step(f"Fold {fold} verileri yükleniyor | {run_name}")
        train_raw, val_raw = load_cv_fold_raw_ready(fold)
        train_model, val_model = load_cv_fold_model_ready(fold)

        sampled_train_ids, sampled_entities, sampled_train_raw = sample_ids_by_store_family(
            train_raw,
            sample_ratio=sample_ratio,
            random_state=random_state
        )
        sampled_val_raw = val_raw.merge(sampled_entities, on=["store_nbr", "family"], how="inner")
        sampled_val_ids = sampled_val_raw["id"].tolist()

        sampled_train_model = filter_model_ready_by_ids(train_model, sampled_train_ids)
        sampled_val_model = filter_model_ready_by_ids(val_model, sampled_val_ids)

        X_train, y_train, feature_cols = split_features_target(sampled_train_model)
        X_val, y_val, _ = split_features_target(sampled_val_model)

        selected_features, importance_df = select_top_k_features(
            X_train, y_train, feature_cols,
            top_k=top_k_features,
            random_state=random_state
        )

        importance_df.to_csv(
            os.path.join(output_dir, f"fold_{fold}_feature_importance.csv"),
            index=False
        )
        pd.DataFrame({"selected_feature": selected_features}).to_csv(
            os.path.join(output_dir, f"fold_{fold}_selected_features.csv"),
            index=False
        )

        log_step(f"Fold {fold} training başlıyor | {run_name}")
        start = timed_block_start()

        model = build_lightgbm_model(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )

        model.fit(X_train[selected_features], y_train)
        preds = model.predict(X_val[selected_features])

        timed_block_end(start, f"Fold {fold} training tamamlandı | {run_name}")

        metrics = regression_metrics(y_val, preds)

        row = {"fold": fold}
        row.update(metrics)
        rows.append(row)

        selected_features_summary.append({
            "fold": fold,
            "selected_feature_count": len(selected_features)
        })

        print(f"\nFold {fold} | {run_name}", flush=True)
        print_metrics(metrics, model_name=f"LightGBM CV Fold {fold}")

    results_df = pd.DataFrame(rows)
    avg_metrics = results_df.drop(columns=["fold"]).mean().to_dict()

    print(f"\nLightGBM Time-Series CV Results ({run_name})", flush=True)
    print(results_df, flush=True)

    print(f"\nAverage CV Metrics ({run_name})", flush=True)
    for k, v in avg_metrics.items():
        print(f"{k:<6}: {v:.4f}", flush=True)

    results_df.to_csv(os.path.join(output_dir, "timeseries_cv_results.csv"), index=False)
    save_metrics_csv(avg_metrics, os.path.join(output_dir, "timeseries_cv_average_metrics.csv"))
    pd.DataFrame(selected_features_summary).to_csv(
        os.path.join(output_dir, "timeseries_cv_selected_feature_counts.csv"),
        index=False
    )

    summary_text = f"""
Model: LightGBM
Run: {run_name}
Cross-Validation: TimeSeriesSplit
Number of Folds: {n_folds}

Sampling:
- Method: store-family based representative sampling
- Sample ratio: {sample_ratio}

Feature Selection:
- Method: ExtraTrees importance ranking
- Top-K Features per Fold: {top_k_features}

Parameters:
- n_estimators = {n_estimators}
- learning_rate = {learning_rate}
- max_depth = {max_depth}
- num_leaves = {num_leaves}
- subsample = {subsample}
- colsample_bytree = {colsample_bytree}
- random_state = {random_state}

Average CV Metrics:
{avg_metrics}
""".strip()

    save_text_summary(summary_text, os.path.join(output_dir, "timeseries_cv_summary.txt"))

    return {
        "run_name": run_name,
        "sample_ratio": sample_ratio,
        "top_k_features": top_k_features,
        "cv_MAE": avg_metrics["MAE"],
        "cv_RMSE": avg_metrics["RMSE"],
        "cv_MAPE": avg_metrics["MAPE"],
        "cv_RMSLE": avg_metrics["RMSLE"],
    }


# =========================================================
# MAIN
# =========================================================

def main():
    params = {
        "sample_ratio": 0.20,
        "random_state": 42,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

    top_k_list = [20, 30, 40]
    n_folds = 3

    print("\n" + "=" * 80)
    print("LIGHTGBM MODEL TRAINING + FEATURE SELECTION + SAMPLING")
    print("=" * 80, flush=True)

    holdout_rows = []
    cv_rows = []

    for top_k in top_k_list:
        print("\n" + "-" * 80)
        print(f"RUN STARTED | top_k = {top_k}")
        print("-" * 80, flush=True)

        holdout_result = evaluate_holdout_lightgbm_single(
            top_k_features=top_k,
            **params
        )
        holdout_rows.append(holdout_result)

        cv_result = evaluate_timeseries_cv_lightgbm_single(
            top_k_features=top_k,
            n_folds=n_folds,
            **params
        )
        cv_rows.append(cv_result)

    model_output_dir = get_model_output_dir(MODEL_NAME)

    holdout_comparison_df = pd.DataFrame(holdout_rows).sort_values("validation_RMSLE")
    cv_comparison_df = pd.DataFrame(cv_rows).sort_values("cv_RMSLE")

    holdout_comparison_df.to_csv(
        os.path.join(model_output_dir, "holdout_comparison_topk.csv"),
        index=False
    )
    cv_comparison_df.to_csv(
        os.path.join(model_output_dir, "timeseries_cv_comparison_topk.csv"),
        index=False
    )

    print("\n" + "=" * 80)
    print("HOLDOUT COMPARISON TABLE")
    print("=" * 80, flush=True)
    print(holdout_comparison_df, flush=True)

    print("\n" + "=" * 80)
    print("TIME-SERIES CV COMPARISON TABLE")
    print("=" * 80, flush=True)
    print(cv_comparison_df, flush=True)

    summary_text = f"""
LightGBM Top-K Feature Comparison Completed.

Sampling:
- store-family representative sampling
- sample ratio = {params['sample_ratio']}

Top-K values tested:
{top_k_list}

Holdout comparison table saved:
- holdout_comparison_topk.csv

CV comparison table saved:
- timeseries_cv_comparison_topk.csv
""".strip()

    save_text_summary(summary_text, os.path.join(model_output_dir, "experiment_summary.txt"))

    print("\n" + "=" * 80)
    print("LIGHTGBM TAMAMLANDI")
    print("=" * 80)
    print("Kaydedilen çıktı klasörü:")
    print(model_output_dir, flush=True)


if __name__ == "__main__":
    main()