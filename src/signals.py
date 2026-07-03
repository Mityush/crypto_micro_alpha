import numpy as np
import pandas as pd
from scipy.signal import windows


def get_causal_weights(window_type: str, window: int) -> np.ndarray:

    if window_type == "triang":
        w = windows.triang(window)

    elif window_type == "hann":
        w = windows.hann(window)

    elif window_type == "hamming":
        w = windows.hamming(window)

    elif window_type == "blackman":
        w = windows.blackman(window)

    elif window_type == "gaussian":
        w = windows.gaussian(window, std=window / 6)

    else:
        raise ValueError(f"Unknown window_type: {window_type}")

    w = np.asarray(w, dtype=float)

    if np.any(w < 0):
        raise ValueError(f"Window has negative weights: {window_type}")

    if w.sum() == 0:
        raise ValueError(f"Window weights sum to zero: {window_type}")

    return w / w.sum()


def rolling_weighted_mean_std(
    s: pd.Series,
    window: int,
    window_type: str = "sma",
) -> tuple[pd.Series, pd.Series]:

    if window_type == "sma":
        mean = s.rolling(window, min_periods=window).mean()
        std = s.rolling(window, min_periods=window).std()
        return mean, std

    if window_type == "exponential":
        mean = s.ewm(span=window, adjust=False, min_periods=window).mean()
        std = s.ewm(span=window, adjust=False, min_periods=window).std()
        return mean, std

    weights = get_causal_weights(window_type, window)

    def wmean(x):
        return np.sum(weights * x)

    def wstd(x):
        mu = np.sum(weights * x)
        var = np.sum(weights * (x - mu) ** 2)
        return np.sqrt(var)

    mean = s.rolling(window, min_periods=window).apply(wmean, raw=True)
    std = s.rolling(window, min_periods=window).apply(wstd, raw=True)

    return mean, std


def mean_reversion_zscore(
    df: pd.DataFrame,
    lookback: int,
    z_window: int,
    mean_window_type: str = "sma",
) -> pd.DataFrame:
    df = df.copy()

    df["lookback_ret"] = (
        df["log_return_1s"]
        .rolling(lookback, min_periods=lookback)
        .sum()
    )

    df["past_ret_mean"], df["past_ret_std"] = rolling_weighted_mean_std(
        df["lookback_ret"],
        window=z_window,
        window_type=mean_window_type,
    )

    df["z_score"] = (
        (df["lookback_ret"] - df["past_ret_mean"])
        / df["past_ret_std"].replace(0, np.nan)
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