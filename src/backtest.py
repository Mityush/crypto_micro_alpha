import numpy as np
import pandas as pd
from src.signals import mean_reversion_zscore, build_threshold_position

def run_mr_backtest(
    df: pd.DataFrame,
    lookback: int,
    z_window: int,
    entry_score: float,
    exit_score: float,
    fee_bps: float = 0.0,
    spread_filter: str = "none",  # none, low, high
    spread_window: int = 6000,
    spread_quantile: float = 0.5,
) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)
    df = mean_reversion_zscore(
        df,
        lookback=lookback,
        z_window=z_window,
    )

    df = build_threshold_position(
        df,
        entry_score=entry_score,
        exit_score=exit_score,
        spread_filter=spread_filter,
        spread_window=spread_window,
        spread_quantile=spread_quantile,
    )

    df["position"] = (
        df["position_raw"]
        .shift(1)
        .fillna(0)
    )

    df["trade_size"] = df["position"].diff().fillna(df["position"])

    df["gross_ret"] = (
        df["position"].shift(1).fillna(0)
        * df["log_return_1s"]
    )

    df["spread_cost"] = (
        df["trade_size"].abs()
        * df["spread_bps"]
        / 20000
    )

    df["fee_cost"] = (
        df["trade_size"].abs()
        * fee_bps
        / 10000
    )

    df["net_ret"] = (
        df["gross_ret"]
        - df["spread_cost"]
        - df["fee_cost"]
    )

    df["equity_gross"] = df["gross_ret"].fillna(0).cumsum()
    df["equity_net"] = df["net_ret"].fillna(0).cumsum()
    df["capital_curve_gross"] = np.exp(df["equity_gross"])
    df["capital_curve_net"] = np.exp(df["equity_net"])

    return df