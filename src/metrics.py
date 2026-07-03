import numpy as np
import pandas as pd

def max_drawdown_capital(capital_curve: pd.Series) -> float:
    capital_curve = capital_curve.ffill().fillna(1.0)
    running_max = capital_curve.cummax()
    drawdown = capital_curve / running_max - 1
    return float(drawdown.min())

def make_metrics(bt: pd.DataFrame) -> dict:
    turnover = bt["trade_size"].abs()
    net_std = bt["net_ret"].std()

    return {
        "total_return_gross": float(bt["capital_curve_gross"].iloc[-1] - 1),
        "total_return_net": float(bt["capital_curve_net"].iloc[-1] - 1),

        "log_gross_pnl": float(bt["gross_ret"].sum()),
        "log_net_pnl": float(bt["net_ret"].sum()),

        "spread_cost": float(bt["spread_cost"].sum()),
        "fee_cost": float(bt["fee_cost"].sum()),
        
        "max_drawdown_gross": max_drawdown_capital(bt["capital_curve_gross"]),
        "max_drawdown_net": max_drawdown_capital(bt["capital_curve_net"]),

        "n_trades": int((turnover > 0).sum()),
        "turnover": float(turnover.sum()),
        "time_in_market": float((bt["position"] != 0).mean()),
        "long_time": float((bt["position"] == 1).mean()),
        "short_time": float((bt["position"] == -1).mean()),

        "mean_net_ret": float(bt["net_ret"].mean()),
        "std_net_ret": float(net_std),
        "sharpe_net": float(bt["net_ret"].mean() / net_std) if net_std > 0 else np.nan,

        "avg_spread_bps": float(bt["spread_bps"].mean()),
        "median_spread_bps": float(bt["spread_bps"].median()),
    }


def make_daily(bt: pd.DataFrame) -> pd.DataFrame:
    df = bt.copy()
    df["date"] = df["datetime"].dt.date

    return (
        df.groupby("date")
        .agg(
            gross_pnl=("gross_ret", "sum"),
            spread_cost=("spread_cost", "sum"),
            fee_cost=("fee_cost", "sum"),
            net_pnl=("net_ret", "sum"),
            turnover=("trade_size", lambda x: x.abs().sum()),
            time_in_market=("position", lambda x: (x != 0).mean()),
        )
        .reset_index()
    )


def make_equity(bt: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "datetime",
        "equity_gross",
        "equity_net",
        "position",
        "z_score",
        "spread_bps",
        "capital_curve_gross",
        "capital_curve_net",
    ]
    return bt[cols].copy()


def make_trades(bt: pd.DataFrame) -> pd.DataFrame:
    trades = []

    in_trade = False
    entry_idx = None
    entry_pos = None

    for idx, pos in enumerate(bt["position"]):
        if (not in_trade) and (pos != 0):
            entry_idx = idx
            entry_pos = pos
            in_trade = True

        elif in_trade and (pos == 0):
            exit_idx = idx

            trade_ret = bt.loc[entry_idx:exit_idx, "net_ret"].sum()
            gross_ret = bt.loc[entry_idx:exit_idx, "gross_ret"].sum()
            spread_cost = bt.loc[entry_idx:exit_idx, "spread_cost"].sum()
            fee_cost = bt.loc[entry_idx:exit_idx, "fee_cost"].sum()

            trades.append({
                "entry_time": bt.loc[entry_idx, "datetime"],
                "exit_time": bt.loc[exit_idx, "datetime"],
                "side": int(entry_pos),
                "holding_time": int(exit_idx - entry_idx),
                "gross_ret": float(gross_ret),
                "spread_cost": float(spread_cost),
                "fee_cost": float(fee_cost),
                "net_ret": float(trade_ret),
            })

            in_trade = False

    return pd.DataFrame(trades)

def make_trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {
            "trade_win_rate": np.nan,
            "long_win_rate": np.nan,
            "short_win_rate": np.nan,
            "n_completed_trades": 0,
            "n_long_trades": 0,
            "n_short_trades": 0,
        }

    long_trades = trades[trades["side"] == 1]
    short_trades = trades[trades["side"] == -1]

    return {
        "trade_win_rate": float((trades["net_ret"] > 0).mean()),
        "long_win_rate": float((long_trades["net_ret"] > 0).mean()) if len(long_trades) > 0 else np.nan,
        "short_win_rate": float((short_trades["net_ret"] > 0).mean()) if len(short_trades) > 0 else np.nan,

        "n_completed_trades": int(len(trades)),
        "n_long_trades": int(len(long_trades)),
        "n_short_trades": int(len(short_trades)),

        "avg_trade_net_ret": float(trades["net_ret"].mean()),
        "avg_long_trade_net_ret": float(long_trades["net_ret"].mean()) if len(long_trades) > 0 else np.nan,
        "avg_short_trade_net_ret": float(short_trades["net_ret"].mean()) if len(short_trades) > 0 else np.nan,
    }