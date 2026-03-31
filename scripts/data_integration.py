import pandas as pd
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(CURRENT_DIR, "..", "data", "raw")


def load_raw_data(base_path=BASE_PATH):
    train = pd.read_csv(os.path.join(base_path, "train.csv"), parse_dates=["date"])
    test = pd.read_csv(os.path.join(base_path, "test.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(base_path, "stores.csv"))
    oil = pd.read_csv(os.path.join(base_path, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(base_path, "holidays_events.csv"), parse_dates=["date"])
    transactions = pd.read_csv(os.path.join(base_path, "transactions.csv"), parse_dates=["date"])
    return train, test, stores, oil, holidays, transactions


def prepare_holiday_features(holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Global holiday tablosu üretir.
    Not: Bu versiyon EDA ve genel preprocessing için uygundur.
    Store-specific holiday mantığı modelleme aşamasında ayrıca eklenebilir.
    """
    h = holidays.copy()

    # Aktarılmış tatilleri çıkar
    h = h[h["transferred"] == False].copy()

    h = h[["date", "type", "locale", "locale_name", "description"]].copy()
    h = h.sort_values("date").drop_duplicates(subset=["date"], keep="first")

    h = h.rename(columns={
        "type": "holiday_type",
        "locale": "holiday_locale",
        "locale_name": "holiday_locale_name",
        "description": "holiday_description",
    })

    h["is_holiday"] = 1
    return h


def merge_all(df: pd.DataFrame,
              stores: pd.DataFrame,
              oil: pd.DataFrame,
              holidays_prepared: pd.DataFrame,
              transactions: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.merge(stores, on="store_nbr", how="left")
    out = out.merge(oil, on="date", how="left")
    out = out.merge(transactions, on=["date", "store_nbr"], how="left")
    out = out.merge(holidays_prepared, on="date", how="left")

    out["is_holiday"] = out["is_holiday"].fillna(0).astype(int)

    return out


def get_integrated_data(base_path=BASE_PATH):
    train, test, stores, oil, holidays, transactions = load_raw_data(base_path)
    holidays_prepared = prepare_holiday_features(holidays)

    train_integrated = merge_all(train, stores, oil, holidays_prepared, transactions)
    test_integrated = merge_all(test, stores, oil, holidays_prepared, transactions)

    return train_integrated, test_integrated