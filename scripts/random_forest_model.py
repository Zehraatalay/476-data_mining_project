import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

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
    save_json
)


MODEL_NAME = "Random Forest"


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


# =========================================================
# FEATURE SELECTION
# =========================================================

def select_top_k_features(X_train, y_train, feature_cols, top_k=40, random_state=42):
    """
    ExtraTrees ile hızlı feature importance çıkarıp top-k seçiyoruz.
    """
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
# HOLDOUT EVALUATION
# =========================================================

def evaluate_holdout_random_forest_single(
    top_k_features,
    n_estimators=150,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
):
    run_name = f"top_{top_k_features}"
    output_dir = os.path.join(get_model_output_dir(MODEL_NAME), run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_step(f"HOLDOUT evaluation başlıyor | {run_name}")

    train_df, val_df, test_df = load_holdout_model_ready()

    # -------------------------------------------------
    # 1) TRAIN -> VALIDATION
    # -------------------------------------------------
    log_step(f"Train -> Validation hazırlığı | {run_name}")
    X_train, y_train, feature_cols = split_features_target(train_df)
    X_val, y_val, _ = split_features_target(val_df)

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

    log_step(f"Random Forest validation training başlıyor | {run_name}")
    start = timed_block_start()

    model_val = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model_val.fit(X_train[selected_features], y_train)
    val_preds = model_val.predict(X_val[selected_features])

    timed_block_end(start, f"Validation training tamamlandı | {run_name}")

    val_metrics = regression_metrics(y_val, val_preds)
    print_metrics(val_metrics, model_name=f"Random Forest Validation Results ({run_name})")

    val_pred_df = val_df[["id", "date", "sales"]].copy()
    val_pred_df["prediction"] = val_preds

    save_metrics_csv(val_metrics, os.path.join(output_dir, "validation_metrics.csv"))
    save_predictions_csv(val_pred_df, os.path.join(output_dir, "validation_predictions.csv"))

    # -------------------------------------------------
    # 2) TRAIN + VALIDATION -> TEST
    # -------------------------------------------------
    log_step(f"Train+Validation -> Test hazırlığı | {run_name}")
    train_plus_val = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    X_train_full, y_train_full, feature_cols_full = split_features_target(train_plus_val)
    X_test, y_test, _ = split_features_target(test_df)

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

    log_step(f"Random Forest final test training başlıyor | {run_name}")
    start = timed_block_start()

    model_test = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model_test.fit(X_train_full[selected_features_test], y_train_full)
    test_preds = model_test.predict(X_test[selected_features_test])

    timed_block_end(start, f"Test training tamamlandı | {run_name}")

    test_metrics = regression_metrics(y_test, test_preds)
    print_metrics(test_metrics, model_name=f"Random Forest Test Results ({run_name})")

    test_pred_df = test_df[["id", "date", "sales"]].copy()
    test_pred_df["prediction"] = test_preds

    save_metrics_csv(test_metrics, os.path.join(output_dir, "test_metrics.csv"))
    save_predictions_csv(test_pred_df, os.path.join(output_dir, "test_predictions.csv"))

    # -------------------------------------------------
    # 3) Summary
    # -------------------------------------------------
    summary_text = f"""
Model: Random Forest Regressor
Run: {run_name}

Feature Selection:
- Method: ExtraTrees importance ranking
- Top-K Features: {top_k_features}

Parameters:
- n_estimators = {n_estimators}
- max_depth = {max_depth}
- min_samples_split = {min_samples_split}
- min_samples_leaf = {min_samples_leaf}
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
            "top_k_features": top_k_features,
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "random_state": random_state
            },
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics
        },
        os.path.join(output_dir, "holdout_summary.json")
    )

    result_row = {
        "run_name": run_name,
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

    return result_row


# =========================================================
# TIME-SERIES CV
# =========================================================

def evaluate_timeseries_cv_random_forest_single(
    top_k_features,
    n_estimators=150,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
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
        train_df, val_df = load_cv_fold_model_ready(fold)

        X_train, y_train, feature_cols = split_features_target(train_df)
        X_val, y_val, _ = split_features_target(val_df)

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

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs
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
        print_metrics(metrics, model_name=f"Random Forest CV Fold {fold}")

    results_df = pd.DataFrame(rows)
    avg_metrics = results_df.drop(columns=["fold"]).mean().to_dict()

    print(f"\nRandom Forest Time-Series CV Results ({run_name})", flush=True)
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
Model: Random Forest Regressor
Run: {run_name}
Cross-Validation: TimeSeriesSplit
Number of Folds: {n_folds}

Feature Selection:
- Method: ExtraTrees importance ranking
- Top-K Features per Fold: {top_k_features}

Parameters:
- n_estimators = {n_estimators}
- max_depth = {max_depth}
- min_samples_split = {min_samples_split}
- min_samples_leaf = {min_samples_leaf}
- random_state = {random_state}

Average CV Metrics:
{avg_metrics}
""".strip()

    save_text_summary(summary_text, os.path.join(output_dir, "timeseries_cv_summary.txt"))

    result_row = {
        "run_name": run_name,
        "top_k_features": top_k_features,
        "cv_MAE": avg_metrics["MAE"],
        "cv_RMSE": avg_metrics["RMSE"],
        "cv_MAPE": avg_metrics["MAPE"],
        "cv_RMSLE": avg_metrics["RMSLE"],
    }

    return result_row


# =========================================================
# KAGGLE SUBMISSION
# =========================================================

def create_kaggle_submission_random_forest(
    top_k_features=40,
    n_estimators=150,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
):
    log_step("Kaggle submission hazırlanıyor")
    output_dir = os.path.join(get_model_output_dir(MODEL_NAME), f"top_{top_k_features}")
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df, _ = load_holdout_model_ready()
    kaggle_test_df = load_kaggle_model_ready()

    train_full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    X_train, y_train, feature_cols = split_features_target(train_full_df)
    selected_features, importance_df = select_top_k_features(
        X_train, y_train, feature_cols,
        top_k=top_k_features,
        random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )

    log_step(f"Kaggle model fit başlıyor | top_{top_k_features}")
    start = timed_block_start()
    model.fit(X_train[selected_features], y_train)
    preds = model.predict(kaggle_test_df[selected_features])
    timed_block_end(start, f"Kaggle model fit tamamlandı | top_{top_k_features}")

    submission = pd.DataFrame({
        "id": kaggle_test_df["id"],
        "sales": preds
    })

    save_submission(submission, filename=f"random_forest_submission_top_{top_k_features}.csv")
    importance_df.to_csv(os.path.join(output_dir, "kaggle_feature_importance.csv"), index=False)


# =========================================================
# MAIN
# =========================================================

def main():
    params = {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": 42,
        "n_jobs": -1
    }

    top_k_list = [20, 30, 40]
    n_folds = 3

    print("\n" + "=" * 80)
    print("RANDOM FOREST MODEL TRAINING + FEATURE SELECTION COMPARISON")
    print("=" * 80, flush=True)

    holdout_rows = []
    cv_rows = []

    for top_k in top_k_list:
        print("\n" + "-" * 80)
        print(f"RUN STARTED | top_k = {top_k}")
        print("-" * 80, flush=True)

        holdout_result = evaluate_holdout_random_forest_single(
            top_k_features=top_k,
            **params
        )
        holdout_rows.append(holdout_result)

        cv_result = evaluate_timeseries_cv_random_forest_single(
            top_k_features=top_k,
            n_folds=n_folds,
            **params
        )
        cv_rows.append(cv_result)

    # -------------------------------------------------
    # Comparison tables
    # -------------------------------------------------
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
Random Forest Top-K Feature Comparison Completed.

Top-K values tested:
{top_k_list}

Holdout comparison table saved:
- holdout_comparison_topk.csv

CV comparison table saved:
- timeseries_cv_comparison_topk.csv
""".strip()

    save_text_summary(summary_text, os.path.join(model_output_dir, "experiment_summary.txt"))

    print("\n" + "=" * 80)
    print("RANDOM FOREST TAMAMLANDI")
    print("=" * 80)
    print("Kaydedilen çıktı klasörü:")
    print(model_output_dir, flush=True)

    # İstersen en iyi top_k için submission aç:
    # create_kaggle_submission_random_forest(top_k_features=40, **params)


if __name__ == "__main__":
    main()