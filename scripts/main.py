import os

from data_integration import get_integrated_data
from data_preprocessing import preprocess_data
from data_understanding import run_eda
from data_splitting import (
    split_by_date,
    create_holdout_summary,
    print_holdout_summary,
    create_time_series_cv_summary,
    print_time_series_cv_summary,
    save_split_outputs,
    save_time_series_cv_folds,
    get_time_series_cv_splits
)
from advanced_preprocessing import ModelingPreprocessor


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    processed_base = os.path.join(base_dir, "..", "data", "processed")
    holdout_path = os.path.join(processed_base, "holdout")
    kaggle_path = os.path.join(processed_base, "kaggle")
    timeseries_cv_path = os.path.join(processed_base, "timeseries_cv")
    split_output_path = os.path.join(base_dir, "..", "outputs", "splits")

    model_ready_path = os.path.join(processed_base, "model_ready")
    holdout_model_ready_path = os.path.join(model_ready_path, "holdout")
    kaggle_model_ready_path = os.path.join(model_ready_path, "kaggle")
    timeseries_cv_model_ready_path = os.path.join(model_ready_path, "timeseries_cv")

    os.makedirs(holdout_path, exist_ok=True)
    os.makedirs(kaggle_path, exist_ok=True)
    os.makedirs(timeseries_cv_path, exist_ok=True)
    os.makedirs(split_output_path, exist_ok=True)
    os.makedirs(holdout_model_ready_path, exist_ok=True)
    os.makedirs(kaggle_model_ready_path, exist_ok=True)
    os.makedirs(timeseries_cv_model_ready_path, exist_ok=True)

    # -------------------------------------------------
    # 1) Integration
    # -------------------------------------------------
    print("1) Veriler birleştiriliyor...")
    train_int, kaggle_test_int = get_integrated_data()

    # -------------------------------------------------
    # 2) Deterministic preprocessing
    # -------------------------------------------------
    print("2) Ön işleme adımları uygulanıyor...")
    train_final, kaggle_test_final = preprocess_data(train_int, kaggle_test_int)

    # -------------------------------------------------
    # 3) Holdout split
    # -------------------------------------------------
    train_end = "2016-01-01"
    val_end = "2017-01-01"

    train_holdout, validation_holdout, test_holdout = split_by_date(
        train_final,
        train_end=train_end,
        val_end=val_end
    )

    # Save deterministic processed datasets
    train_holdout.to_csv(os.path.join(holdout_path, "train_holdout_ready.csv"), index=False)
    validation_holdout.to_csv(os.path.join(holdout_path, "validation_holdout_ready.csv"), index=False)
    test_holdout.to_csv(os.path.join(holdout_path, "test_holdout_ready.csv"), index=False)
    kaggle_test_final.to_csv(os.path.join(kaggle_path, "kaggle_test_ready.csv"), index=False)

    print("\nProcessed dosyalar kaydedildi:")
    print(f"- {os.path.join(holdout_path, 'train_holdout_ready.csv')}")
    print(f"- {os.path.join(holdout_path, 'validation_holdout_ready.csv')}")
    print(f"- {os.path.join(holdout_path, 'test_holdout_ready.csv')}")
    print(f"- {os.path.join(kaggle_path, 'kaggle_test_ready.csv')}")

    # -------------------------------------------------
    # 4) EDA
    # -------------------------------------------------
    print("\n3) EDA çalıştırılıyor...")
    run_eda(train_final, dataset_name="FINAL TRAIN")

    # -------------------------------------------------
    # 5) Holdout summary
    # -------------------------------------------------
    holdout_summary_df = create_holdout_summary(
        train_holdout,
        validation_holdout,
        test_holdout,
        train_end=train_end,
        val_end=val_end
    )
    print_holdout_summary(holdout_summary_df)

    # -------------------------------------------------
    # 6) Time-series Cross-Validation
    # CV pool = train + validation period only
    # -------------------------------------------------
    print("\n4) Time-series cross-validation hazırlanıyor...")
    cv_pool = train_final[train_final["date"] < val_end].copy()

    cv_summary_df = create_time_series_cv_summary(cv_pool, n_splits=5)
    print_time_series_cv_summary(cv_summary_df)

    save_split_outputs(
        holdout_summary_df=holdout_summary_df,
        cv_summary_df=cv_summary_df,
        output_dir=split_output_path
    )

    save_time_series_cv_folds(
        df=cv_pool,
        output_dir=timeseries_cv_path,
        n_splits=5
    )

    # -------------------------------------------------
    # 7) Modeling-ready holdout datasets
    # -------------------------------------------------
    print("\n5) Leakage-safe model-ready veri üretiliyor...")

    # Train -> Validation
    holdout_preprocessor = ModelingPreprocessor()
    train_holdout_model = holdout_preprocessor.fit_transform(train_holdout)
    validation_holdout_model = holdout_preprocessor.transform(validation_holdout)

    train_holdout_model.to_csv(
        os.path.join(holdout_model_ready_path, "train_holdout_model_ready.csv"),
        index=False
    )
    validation_holdout_model.to_csv(
        os.path.join(holdout_model_ready_path, "validation_holdout_model_ready.csv"),
        index=False
    )

    # Train+Validation -> Test
    train_val_concat = train_final[train_final["date"] < val_end].copy()
    test_preprocessor = ModelingPreprocessor()
    _ = test_preprocessor.fit_transform(train_val_concat)
    test_holdout_model = test_preprocessor.transform(test_holdout)

    test_holdout_model.to_csv(
        os.path.join(holdout_model_ready_path, "test_holdout_model_ready.csv"),
        index=False
    )

    # -------------------------------------------------
    # 8) Modeling-ready Kaggle test
    # Final training pool = full train_final
    # -------------------------------------------------
    kaggle_preprocessor = ModelingPreprocessor()
    _ = kaggle_preprocessor.fit_transform(train_final)
    kaggle_test_model = kaggle_preprocessor.transform(kaggle_test_final)

    kaggle_test_model.to_csv(
        os.path.join(kaggle_model_ready_path, "kaggle_test_model_ready.csv"),
        index=False
    )

    # -------------------------------------------------
    # 9) Modeling-ready CV folds
    # -------------------------------------------------
    cv_splits = get_time_series_cv_splits(cv_pool, n_splits=5)

    for split in cv_splits:
        fold = split["fold"]
        fold_train = split["train_df"]
        fold_val = split["val_df"]

        fold_preprocessor = ModelingPreprocessor()
        fold_train_model = fold_preprocessor.fit_transform(fold_train)
        fold_val_model = fold_preprocessor.transform(fold_val)

        fold_train_model.to_csv(
            os.path.join(timeseries_cv_model_ready_path, f"fold_{fold}_train_model_ready.csv"),
            index=False
        )
        fold_val_model.to_csv(
            os.path.join(timeseries_cv_model_ready_path, f"fold_{fold}_validation_model_ready.csv"),
            index=False
        )

    print("\nModel-ready dosyalar kaydedildi:")
    print(f"- {holdout_model_ready_path}")
    print(f"- {kaggle_model_ready_path}")
    print(f"- {timeseries_cv_model_ready_path}")

    print("\n[BAŞARILI] Veri madenciliği boru hattı akademik olarak temiz biçimde modeling-ready aşamasına ulaştı.")


if __name__ == "__main__":
    main()