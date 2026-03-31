import os
import time
import itertools
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from modeling_utils import (
    load_holdout_model_ready,
    split_features_target,
    regression_metrics,
    print_metrics,
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


MODEL_NAME = "XGBoost_Tuned"


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

def build_xgboost_model(params, random_state=42):
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",         # hızlı
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **params
    )


# =========================================================
# PARAM GRID
# =========================================================

def get_param_grid():
    grid = {
        "n_estimators": [300, 500],
        "learning_rate": [0.03, 0.05],
        "max_depth": [6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
        "gamma": [0.0, 0.1],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 2.0],
    }

    keys = list(grid.keys())
    values = list(grid.values())

    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# =========================================================
# TUNING
# =========================================================

def tune_xgboost_on_validation(
    train_df_model,
    val_df_model,
    top_k_features,
    output_dir,
    random_state=42
):
    X_train, y_train, feature_cols = split_features_target(train_df_model)
    X_val, y_val, _ = split_features_target(val_df_model)

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

    param_grid = get_param_grid()
    tuning_rows = []

    best_model = None
    best_params = None
    best_metrics = None
    best_val_preds = None
    best_score = float("inf")

    log_step(f"Hyperparameter tuning başlıyor | toplam kombinasyon: {len(param_grid)}")

    for i, params in enumerate(param_grid, start=1):
        print(f"\n[TUNING] Combination {i}/{len(param_grid)}", flush=True)
        print(params, flush=True)

        start = timed_block_start()

        model = build_xgboost_model(params, random_state=random_state)
        model.fit(
            X_train[selected_features],
            y_train,
            eval_set=[(X_val[selected_features], y_val)],
            verbose=False
        )

        val_preds = model.predict(X_val[selected_features])
        val_metrics = regression_metrics(y_val, val_preds)

        timed_block_end(start, f"Tuning combination {i} tamamlandı")
        print_metrics(val_metrics, model_name=f"Validation Metrics | Combo {i}")

        row = {"combo_idx": i, "top_k_features": top_k_features}
        row.update(params)
        row.update({
            "validation_MAE": val_metrics["MAE"],
            "validation_RMSE": val_metrics["RMSE"],
            "validation_MAPE": val_metrics["MAPE"],
            "validation_RMSLE": val_metrics["RMSLE"],
        })
        tuning_rows.append(row)

        if val_metrics["RMSLE"] < best_score:
            best_score = val_metrics["RMSLE"]
            best_model = model
            best_params = params
            best_metrics = val_metrics
            best_val_preds = val_preds

    tuning_df = pd.DataFrame(tuning_rows).sort_values("validation_RMSLE")
    tuning_df.to_csv(os.path.join(output_dir, "tuning_results_validation.csv"), index=False)

    best_val_pred_df = val_df_model[["id", "date", "sales"]].copy()
    best_val_pred_df["prediction"] = best_val_preds
    save_predictions_csv(best_val_pred_df, os.path.join(output_dir, "best_validation_predictions.csv"))
    save_metrics_csv(best_metrics, os.path.join(output_dir, "best_validation_metrics.csv"))

    return {
        "selected_features": selected_features,
        "best_params": best_params,
        "best_metrics": best_metrics,
        "tuning_df": tuning_df
    }


# =========================================================
# FINAL TEST
# =========================================================

def evaluate_best_model_on_test(
    train_plus_val_model,
    test_model,
    top_k_features,
    best_params,
    output_dir,
    random_state=42
):
    X_train_full, y_train_full, feature_cols = split_features_target(train_plus_val_model)
    X_test, y_test, _ = split_features_target(test_model)

    selected_features_test, importance_df_test = select_top_k_features(
        X_train_full, y_train_full, feature_cols,
        top_k=top_k_features,
        random_state=random_state
    )

    importance_df_test.to_csv(os.path.join(output_dir, "feature_importance_train_val.csv"), index=False)
    pd.DataFrame({"selected_feature": selected_features_test}).to_csv(
        os.path.join(output_dir, "selected_features_test_stage.csv"),
        index=False
    )

    log_step("Final model training (train+validation) başlıyor")
    start = timed_block_start()

    model = build_xgboost_model(best_params, random_state=random_state)
    model.fit(X_train_full[selected_features_test], y_train_full)

    test_preds = model.predict(X_test[selected_features_test])

    timed_block_end(start, "Final model training + test prediction tamamlandı")

    test_metrics = regression_metrics(y_test, test_preds)
    print_metrics(test_metrics, model_name="Best XGBoost Test Results")

    test_pred_df = test_model[["id", "date", "sales"]].copy()
    test_pred_df["prediction"] = test_preds

    save_metrics_csv(test_metrics, os.path.join(output_dir, "best_test_metrics.csv"))
    save_predictions_csv(test_pred_df, os.path.join(output_dir, "best_test_predictions.csv"))

    return {
        "test_metrics": test_metrics,
        "selected_features_test": selected_features_test
    }


# =========================================================
# SINGLE RUN
# =========================================================

def run_single_experiment(
    top_k_features,
    sample_ratio=0.20,
    random_state=42
):
    run_name = f"top_{top_k_features}"
    output_dir = os.path.join(get_output_dir(), run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_step(f"EXPERIMENT BAŞLADI | {run_name}")

    train_raw, val_raw, test_raw = load_holdout_raw_ready()
    train_model, val_model, test_model = load_holdout_model_ready()

    # raw üzerinden sampling
    log_step(f"Train holdout örnekleniyor | ratio={sample_ratio}")
    sampled_train_ids, sampled_entities, sampled_train_raw = sample_ids_by_store_family(
        train_raw,
        sample_ratio=sample_ratio,
        random_state=random_state
    )
    print_sampling_summary(train_raw, sampled_train_raw, sampled_entities)

    sampled_entities.to_csv(os.path.join(output_dir, "sampled_entities_train.csv"), index=False)

    sampled_val_raw = val_raw.merge(sampled_entities, on=["store_nbr", "family"], how="inner")
    sampled_test_raw = test_raw.merge(sampled_entities, on=["store_nbr", "family"], how="inner")

    sampled_val_ids = sampled_val_raw["id"].tolist()
    sampled_test_ids = sampled_test_raw["id"].tolist()

    sampled_train_model = filter_model_ready_by_ids(train_model, sampled_train_ids)
    sampled_val_model = filter_model_ready_by_ids(val_model, sampled_val_ids)
    sampled_test_model = filter_model_ready_by_ids(test_model, sampled_test_ids)

    # validation tuning
    tuning_result = tune_xgboost_on_validation(
        sampled_train_model,
        sampled_val_model,
        top_k_features=top_k_features,
        output_dir=output_dir,
        random_state=random_state
    )

    # final test
    sampled_train_plus_val_model = pd.concat(
        [sampled_train_model, sampled_val_model],
        axis=0
    ).reset_index(drop=True)

    test_result = evaluate_best_model_on_test(
        sampled_train_plus_val_model,
        sampled_test_model,
        top_k_features=top_k_features,
        best_params=tuning_result["best_params"],
        output_dir=output_dir,
        random_state=random_state
    )

    summary_text = f"""
Model: XGBoost Tuned
Run: {run_name}

Sampling:
- Method: store-family based representative sampling
- Sample ratio: {sample_ratio}

Feature Selection:
- Method: ExtraTrees importance ranking
- Top-K Features: {top_k_features}

Best Validation Params:
{tuning_result['best_params']}

Best Validation Metrics:
{tuning_result['best_metrics']}

Final Test Metrics:
{test_result['test_metrics']}
""".strip()

    save_text_summary(summary_text, os.path.join(output_dir, "experiment_summary.txt"))
    save_json(
        {
            "model": MODEL_NAME,
            "run_name": run_name,
            "sample_ratio": sample_ratio,
            "top_k_features": top_k_features,
            "best_params": tuning_result["best_params"],
            "best_validation_metrics": tuning_result["best_metrics"],
            "final_test_metrics": test_result["test_metrics"],
        },
        os.path.join(output_dir, "experiment_summary.json")
    )

    return {
        "run_name": run_name,
        "sample_ratio": sample_ratio,
        "top_k_features": top_k_features,
        "best_validation_RMSLE": tuning_result["best_metrics"]["RMSLE"],
        "best_test_RMSLE": test_result["test_metrics"]["RMSLE"],
        "best_test_RMSE": test_result["test_metrics"]["RMSE"],
        "best_test_MAE": test_result["test_metrics"]["MAE"],
    }


# =========================================================
# MAIN
# =========================================================

def main():
    print("\n" + "=" * 80)
    print("XGBOOST TUNED MODEL TRAINING")
    print("=" * 80, flush=True)

    top_k_list = [20, 30, 40]
    summary_rows = []

    for top_k in top_k_list:
        print("\n" + "-" * 80)
        print(f"TUNED RUN STARTED | top_k = {top_k}")
        print("-" * 80, flush=True)

        result = run_single_experiment(
            top_k_features=top_k,
            sample_ratio=0.20,
            random_state=42
        )
        summary_rows.append(result)

    summary_df = pd.DataFrame(summary_rows).sort_values("best_validation_RMSLE")
    model_output_dir = get_output_dir()

    summary_df.to_csv(os.path.join(model_output_dir, "topk_tuning_summary.csv"), index=False)

    print("\n" + "=" * 80)
    print("FINAL TUNING SUMMARY")
    print("=" * 80, flush=True)
    print(summary_df, flush=True)

    print("\n" + "=" * 80)
    print("XGBOOST TUNED TAMAMLANDI")
    print("=" * 80)
    print("Kaydedilen çıktı klasörü:")
    print(model_output_dir, flush=True)


if __name__ == "__main__":
    main()