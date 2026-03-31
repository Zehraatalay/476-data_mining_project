import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_eda(df: pd.DataFrame, dataset_name: str = "TRAIN"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, "..", "outputs", "eda", dataset_name.lower().replace(" ", "_"))
    os.makedirs(output_path, exist_ok=True)

    print(f"\nAnaliz sonuçları ve grafikler buraya kaydediliyor: {output_path}")

    # -------------------------------------------------
    # 1) Dataset summary
    # -------------------------------------------------
    summary_lines = []
    summary_lines.append(f"Dataset Name: {dataset_name}")
    summary_lines.append(f"Total Records: {len(df)}")
    summary_lines.append(f"Total Features: {len(df.columns)}")

    if "date" in df.columns:
        summary_lines.append(f"Date Range: {df['date'].min()} -> {df['date'].max()}")
    if "family" in df.columns:
        summary_lines.append(f"Unique Product Families: {df['family'].nunique()}")
    if "store_nbr" in df.columns:
        summary_lines.append(f"Unique Stores: {df['store_nbr'].nunique()}")
    if "city" in df.columns:
        summary_lines.append(f"Unique Cities: {df['city'].nunique()}")
    if "state" in df.columns:
        summary_lines.append(f"Unique States: {df['state'].nunique()}")

    if "sales" in df.columns:
        summary_lines.append(f"Zero Sales Ratio: {(df['sales'] == 0).mean():.4f}")

    summary_text = "\n".join(summary_lines)

    with open(os.path.join(output_path, "dataset_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)

    # -------------------------------------------------
    # 2) Descriptive statistics
    # -------------------------------------------------
    numeric_candidates = ["sales", "onpromotion", "dcoilwtico", "transactions"]
    available_numeric = [col for col in numeric_candidates if col in df.columns]

    if available_numeric:
        stats = df[available_numeric].describe()
        stats.to_csv(os.path.join(output_path, "descriptive_stats.csv"))
        print("\nDescriptive statistics kaydedildi.")

    # -------------------------------------------------
    # 3) Missingness
    # -------------------------------------------------
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["column", "missing_count"]
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
    missing_df.to_csv(os.path.join(output_path, "missingness_summary.csv"), index=False)
    print("Missingness summary kaydedildi.")

    # -------------------------------------------------
    # 4) Sparsity analysis
    # -------------------------------------------------
    if "sales" in df.columns:
        sparsity_summary = pd.DataFrame({
            "metric": [
                "zero_sales_ratio",
                "nonzero_sales_ratio",
                "mean_sales",
                "median_sales",
                "max_sales"
            ],
            "value": [
                (df["sales"] == 0).mean(),
                (df["sales"] != 0).mean(),
                df["sales"].mean(),
                df["sales"].median(),
                df["sales"].max()
            ]
        })
        sparsity_summary.to_csv(os.path.join(output_path, "sparsity_summary.csv"), index=False)

    # -------------------------------------------------
    # 5) Outlier summary (IQR based)
    # -------------------------------------------------
    if "sales" in df.columns:
        q1 = df["sales"].quantile(0.25)
        q3 = df["sales"].quantile(0.75)
        iqr = q3 - q1
        lower = max(0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr

        outlier_flag = ((df["sales"] < lower) | (df["sales"] > upper)).astype(int)
        outlier_summary = pd.DataFrame({
            "q1": [q1],
            "q3": [q3],
            "iqr": [iqr],
            "lower_bound": [lower],
            "upper_bound": [upper],
            "outlier_count": [int(outlier_flag.sum())],
            "outlier_ratio": [float(outlier_flag.mean())]
        })
        outlier_summary.to_csv(os.path.join(output_path, "outlier_summary.csv"), index=False)

    # -------------------------------------------------
    # 6) Sales trends
    # -------------------------------------------------
    if "date" in df.columns and "sales" in df.columns:
        plt.figure(figsize=(12, 6))
        df.groupby("date")["sales"].sum().plot()
        plt.title(f"{dataset_name} - Günlük Toplam Satış Trendi")
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "sales_trend_total.png"))
        plt.close()

        plt.figure(figsize=(12, 6))
        df.groupby("date")["sales"].mean().plot()
        plt.title(f"{dataset_name} - Günlük Ortalama Satış Trendi")
        plt.xlabel("Date")
        plt.ylabel("Mean Sales")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "sales_trend_mean.png"))
        plt.close()

    # -------------------------------------------------
    # 7) Correlation heatmap
    # -------------------------------------------------
    numeric_df = df.select_dtypes(include=["float64", "int64", "int32"]).copy()

    if "id" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["id"])

    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(12, 9))
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
        plt.title(f"{dataset_name} - Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "correlation_heatmap.png"))
        plt.close()

    # -------------------------------------------------
    # 8) Store type sales boxplot
    # -------------------------------------------------
    if "type" in df.columns and "sales" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="type", y="sales", data=df, showfliers=False)
        plt.yscale("log")
        plt.title(f"{dataset_name} - Store Type vs Sales")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "store_type_sales_boxplot.png"))
        plt.close()

    # -------------------------------------------------
    # 9) Weekly pattern
    # -------------------------------------------------
    if "day_of_week" in df.columns and "sales" in df.columns:
        plt.figure(figsize=(8, 5))
        df.groupby("day_of_week")["sales"].mean().plot(kind="bar")
        plt.title(f"{dataset_name} - Weekly Sales Pattern")
        plt.xlabel("Day of Week")
        plt.ylabel("Mean Sales")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "weekly_sales_pattern.png"))
        plt.close()

    print(f"--- Tüm grafikler ve tablolar '{output_path}' klasörüne kaydedildi. ---")