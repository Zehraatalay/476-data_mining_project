import pandas as pd


def sample_ids_by_store_family(df_raw: pd.DataFrame,
                               sample_ratio: float = 0.20,
                               random_state: int = 42):
    """
    Raw holdout dataset üzerinden store_nbr + family bazlı temsilci örneklem alır.
    Dönen sampled_ids, model_ready setleri filtrelemek için kullanılır.
    """
    required_cols = ["id", "store_nbr", "family", "date"]
    missing_cols = [c for c in required_cols if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Sampling için eksik kolonlar var: {missing_cols}")

    entity_df = df_raw[["store_nbr", "family"]].drop_duplicates().copy()

    sampled_entities = entity_df.sample(
        frac=sample_ratio,
        random_state=random_state
    ).reset_index(drop=True)

    sampled_raw = df_raw.merge(
        sampled_entities,
        on=["store_nbr", "family"],
        how="inner"
    ).copy()

    sampled_raw = sampled_raw.sort_values("date").reset_index(drop=True)
    sampled_ids = sampled_raw["id"].tolist()

    return sampled_ids, sampled_entities, sampled_raw


def filter_model_ready_by_ids(df_model_ready: pd.DataFrame, sampled_ids):
    """
    Model-ready dataframe'i sampled id listesine göre filtreler.
    """
    out = df_model_ready[df_model_ready["id"].isin(sampled_ids)].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def print_sampling_summary(original_df: pd.DataFrame,
                           sampled_df: pd.DataFrame,
                           sampled_entities: pd.DataFrame):
    orig_rows = len(original_df)
    sampled_rows = len(sampled_df)

    orig_entities = original_df[["store_nbr", "family"]].drop_duplicates().shape[0]
    sampled_entity_count = len(sampled_entities)

    print("\n--- SAMPLING SUMMARY ---", flush=True)
    print(f"Original rows            : {orig_rows}", flush=True)
    print(f"Sampled rows             : {sampled_rows}", flush=True)
    print(f"Row ratio                : {sampled_rows / orig_rows:.2%}", flush=True)
    print(f"Original store-family    : {orig_entities}", flush=True)
    print(f"Sampled store-family     : {sampled_entity_count}", flush=True)
    print(f"Entity ratio             : {sampled_entity_count / orig_entities:.2%}", flush=True)

    if "date" in sampled_df.columns:
        print(f"Sample date range        : {sampled_df['date'].min()} -> {sampled_df['date'].max()}", flush=True)