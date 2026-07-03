import numpy as np
import pandas as pd


def mean_reversion_zscore(
    df: pd.DataFrame,
    lookback: int,
    z_window: int,
) -> pd.DataFrame:
    df = df.copy()

    df["lookback_ret"] = (
        df["log_return_1s"]
        .rolling(lookback, min_periods=lookback)
        .sum()
    )

    df["past_ret_mean"] = (
        df["lookback_ret"]
        .rolling(z_window, min_periods=z_window)
        .mean()
    )

    df["past_ret_std"] = (
        df["lookback_ret"]
        .rolling(z_window, min_periods=z_window)
        .std()
    )

    df["z_score"] = (
        (df["lookback_ret"] - df["past_ret_mean"])
        / df["past_ret_std"]
    )

    return df


def build_threshold_position(
    df: pd.DataFrame,
    entry_score: float,
    exit_score: float,
    spread_filter: str = "none",
    spread_window: int = 6000,
    spread_quantile: float = 0.5,
) -> pd.DataFrame:
    df = df.copy()

    if spread_filter != "none":
        df["spread_filter_level"] = (
            df["spread_bps"]
            .rolling(spread_window, min_periods=spread_window)
            .quantile(spread_quantile)
            .shift(1)
        )

        if spread_filter == "low":
            df["spread_ok"] = df["spread_bps"] < df["spread_filter_level"]
        elif spread_filter == "high":
            df["spread_ok"] = df["spread_bps"] > df["spread_filter_level"]
        else:
            raise ValueError("spread_filter must be one of: none, low, high")
    else:
        df["spread_ok"] = True

    pos = 0
    positions = []

    for z, spread_ok in zip(df["z_score"], df["spread_ok"]):
        if np.isnan(z) or pd.isna(spread_ok):
            positions.append(0)
            continue

        if pos == 0:
            if spread_ok:
                if z < -entry_score:
                    pos = 1
                elif z > entry_score:
                    pos = -1

        elif pos == 1:
            if z > -exit_score:
                pos = 0

        elif pos == -1:
            if z < exit_score:
                pos = 0

        positions.append(pos)

    df["position_raw"] = positions

    return df