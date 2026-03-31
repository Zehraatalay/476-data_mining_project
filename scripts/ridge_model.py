import os
import pandas as pd
from sklearn.linear_model import Ridge

from modeling_utils import (
    load_holdout_model_ready,
    load_cv_fold_model_ready,
    split_features_target,
    regression_metrics,
    print_metrics,
    fit_and_predict_kaggle,
    load_kaggle_model_ready,
    save_submission,
    get_model_output_dir,
    save_metrics_csv,
    save_predictions_csv,
    save_text_summary,
    save_json
)


MODEL_NAME = "Ridge"


def evaluate_holdout_ridge(alpha=1.0):
    print("\n=== RIDGE: HOLDOUT EVALUATION ===")

    output_dir = get_model_output_dir(MODEL_NAME)

    train_df, val_df, test_df = load_holdout_model_ready()

    # -------------------------------------------------
    # 1) TRAIN -> VALIDATION
    # -------------------------------------------------
    X_train, y_train, feature_cols = split_features_target(train_df)
    X_val, y_val, _ = split_features_target(val_df)

    model_val = Ridge(alpha=alpha)
    model_val.fit(X_train, y_train)
    val_preds = model_val.predict(X_val)

    val_metrics = regression_metrics(y_val, val_preds)
    print_metrics(val_metrics, model_name="Ridge Validation Results")

    val_pred_df = val_df[["id", "date", "sales"]].copy()
    val_pred_df["prediction"] = val_preds

    save_metrics_csv(val_metrics, os.path.join(output_dir, "validation_metrics.csv"))
    save_predictions_csv(val_pred_df, os.path.join(output_dir, "validation_predictions.csv"))

    # -------------------------------------------------
    # 2) TRAIN + VALIDATION -> TEST
    # -------------------------------------------------
    train_plus_val = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    X_train_full, y_train_full, _ = split_features_target(train_plus_val)
    X_test, y_test, _ = split_features_target(test_df)

    model_test = Ridge(alpha=alpha)
    model_test.fit(X_train_full, y_train_full)
    test_preds = model_test.predict(X_test)

    test_metrics = regression_metrics(y_test, test_preds)
    print_metrics(test_metrics, model_name="Ridge Test Results")

    test_pred_df = test_df[["id", "date", "sales"]].copy()
    test_pred_df["prediction"] = test_preds

    save_metrics_csv(test_metrics, os.path.join(output_dir, "test_metrics.csv"))
    save_predictions_csv(test_pred_df, os.path.join(output_dir, "test_predictions.csv"))

    # -------------------------------------------------
    # 3) Summary
    # -------------------------------------------------
    summary_text = f"""
Model: Ridge Regression
Alpha: {alpha}

Validation Strategy:
- Train set used for fitting
- Validation set used for model selection / overfitting control

Final Test Strategy:
- Train + Validation merged for final fitting
- Test set used for final unbiased evaluation

Feature Count: {len(feature_cols)}

Validation Metrics:
{val_metrics}

Test Metrics:
{test_metrics}
""".strip()

    save_text_summary(summary_text, os.path.join(output_dir, "holdout_summary.txt"))
    save_json(
        {
            "model": MODEL_NAME,
            "alpha": alpha,
            "feature_count": len(feature_cols),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics
        },
        os.path.join(output_dir, "holdout_summary.json")
    )

    return {
        "alpha": alpha,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_count": len(feature_cols)
    }


def evaluate_timeseries_cv_ridge(alpha=1.0, n_folds=5):
    print("\n=== RIDGE: TIME-SERIES CV ===")

    output_dir = get_model_output_dir(MODEL_NAME)
    rows = []

    for fold in range(1, n_folds + 1):
        train_df, val_df = load_cv_fold_model_ready(fold)

        X_train, y_train, _ = split_features_target(train_df)
        X_val, y_val, _ = split_features_target(val_df)

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        metrics = regression_metrics(y_val, preds)

        row = {"fold": fold}
        row.update(metrics)
        rows.append(row)

        print(f"\nFold {fold}")
        print_metrics(metrics, model_name=f"Ridge CV Fold {fold}")

    results_df = pd.DataFrame(rows)
    avg_metrics = results_df.drop(columns=["fold"]).mean().to_dict()

    print("\nRidge Time-Series CV Results")
    print(results_df)

    print("\nAverage CV Metrics")
    for k, v in avg_metrics.items():
        print(f"{k:<6}: {v:.4f}")

    results_df.to_csv(os.path.join(output_dir, "timeseries_cv_results.csv"), index=False)
    save_metrics_csv(avg_metrics, os.path.join(output_dir, "timeseries_cv_average_metrics.csv"))

    summary_text = f"""
Model: Ridge Regression
Alpha: {alpha}
Cross-Validation: TimeSeriesSplit
Number of Folds: {n_folds}

Average CV Metrics:
{avg_metrics}
""".strip()

    save_text_summary(summary_text, os.path.join(output_dir, "timeseries_cv_summary.txt"))

    return results_df


def create_kaggle_submission_ridge(alpha=1.0):
    print("\n=== RIDGE: KAGGLE SUBMISSION ===")

    train_df, val_df, _ = load_holdout_model_ready()
    kaggle_test_df = load_kaggle_model_ready()

    train_full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    model = Ridge(alpha=alpha)
    submission = fit_and_predict_kaggle(model, train_full_df, kaggle_test_df)

    save_submission(submission, filename="ridge_submission.csv")


def main():
    alpha = 1.0

    print("\n" + "=" * 70)
    print("RIDGE REGRESSION MODEL TRAINING")
    print("=" * 70)

    holdout_results = evaluate_holdout_ridge(alpha=alpha)
    cv_results = evaluate_timeseries_cv_ridge(alpha=alpha, n_folds=5)

    print("\n" + "=" * 70)
    print("RIDGE TAMAMLANDI")
    print("=" * 70)
    print("Kaydedilen çıktı klasörü:")
    print(get_model_output_dir(MODEL_NAME))

    # İstersen submission da aç:
    # create_kaggle_submission_ridge(alpha=alpha)


if __name__ == "__main__":
    main()