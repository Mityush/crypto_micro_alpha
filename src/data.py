from pathlib import Path
import pandas as pd
import numpy as np


def load_symbol_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    required = [
        "exchange_ts",
        "ask_price_0",
        "bid_price_0",
        "ask_volume_0",
        "bid_volume_0",
        "midprice"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["exchange_ts"], unit="ms")
    df = df.sort_values("datetime").reset_index(drop=True)

    df["spread"] = df["ask_price_0"] - df["bid_price_0"]
    df["spread_bps"] = df["spread"] / df["midprice"] * 10000

    df["log_mid"] = np.log(df["midprice"])
    df["log_return_1s"] = df["log_mid"].diff()

    return df


def load_data(data_dir: str | Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    dfs = {}

    for symbol in symbols:
        candidates = [
            data_dir / f"{symbol}.pkl",
            data_dir / f"{symbol}.parquet",
            data_dir / f"{symbol}.csv",
            data_dir / f"{symbol.lower()}.pkl",
            data_dir / f"{symbol.lower()}.parquet",
            data_dir / f"{symbol.lower()}.csv",
        ]

        found = None
        for p in candidates:
            if p.exists():
                found = p
                break

        if found is None:
            raise FileNotFoundError(f"No file found for symbol={symbol} in {data_dir}")

        dfs[symbol] = load_symbol_file(found)

    return dfs