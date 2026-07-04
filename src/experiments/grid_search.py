import argparse
from pathlib import Path
from itertools import product
import yaml

import pandas as pd
from tqdm import tqdm

from src.data import load_data
from src.backtest import run_mr_backtest
from src.metrics import make_metrics, make_daily, make_equity, make_trades, make_trade_metrics
from src.logger import ResearchLogger



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config",
    )

    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dfs = load_data(cfg["data_dir"], cfg["symbols"])

    logger = ResearchLogger(
        out_dir=cfg["out_dir"],
        project=cfg.get("project", "crypto_micro_alpha"),
        use_wandb=cfg.get("use_wandb", False),
    )

    strategy_name = cfg.get("strategy", "mean_reversion_bid_ask")
    fee_bps = cfg.get("fee_bps", 0.0)

    grid_cfg = cfg["grid"]

    lookbacks = grid_cfg["lookbacks"]
    entries = grid_cfg["entries"]
    exits = grid_cfg["exits"]
    spread_filters = grid_cfg.get("spread_filters")
    spread_quantiles = grid_cfg.get("spread_quantiles")
    mean_window_types = grid_cfg.get("mean_window_types")
    signal_inputs = grid_cfg.get(
        "signal_inputs",
        ["log_return"]
    )

    vol_windows = grid_cfg.get(
        "vol_windows",
        [1800]
    )

    all_runs = []

    grid = list(product(
    cfg["symbols"],
    lookbacks,
    entries,
    exits,
    spread_filters,
    spread_quantiles,
    mean_window_types,
    signal_inputs,
    vol_windows,
    ))

    for symbol, lookback, entry_score, exit_score, spread_filter, spread_quantile, mean_window_type, signal_input, vol_window in tqdm(grid):
        if exit_score >= entry_score:
            continue

        z_window = max(1800, lookback * 10)
        spread_window = z_window

        config = {
            "strategy": strategy_name,
            "symbol": symbol,
            "lookback": lookback,
            "z_window": z_window,
            "entry_score": entry_score,
            "exit_score": exit_score,
            "fee_bps": fee_bps,
            "spread_filter": spread_filter,
            "spread_window": spread_window,
            "spread_quantile": spread_quantile,
            "mean_window_type": mean_window_type,
            "signal_input": signal_input,
            "vol_window": vol_window,
        }

        bt = run_mr_backtest(
            dfs[symbol],
            lookback=lookback,
            z_window=z_window,
            entry_score=entry_score,
            exit_score=exit_score,
            fee_bps=fee_bps,
            spread_filter=spread_filter,
            spread_window=spread_window,
            spread_quantile=spread_quantile,
            mean_window_type=mean_window_type,
            signal_input=signal_input,
            vol_window=vol_window,
        )

        metrics = make_metrics(bt)
        daily = make_daily(bt)
        equity = make_equity(bt)
        trades = make_trades(bt)

        trade_metrics = make_trade_metrics(trades)
        metrics = {**metrics, **trade_metrics}

        run_id = logger.log_run(
            config=config,
            metrics=metrics,
            equity_df=equity,
            daily_df=daily,
            trades_df=trades,
        )

        all_runs.append({"run_id": run_id, **config, **metrics})


    if len(all_runs) == 0:
        print("No successful runs.")
        return

    summary = pd.DataFrame(all_runs).sort_values(
        "total_return_net",
        ascending=False
    )

    preferred_cols = [
        "run_id",
        "strategy",
        "symbol",

        "total_return_net",
        "total_return_gross",
        "log_net_pnl",
        "log_gross_pnl",
        "max_drawdown_net",
        "sharpe_net",

        "trade_win_rate",
        "long_win_rate",
        "short_win_rate",
        "avg_trade_net_ret",

        "n_completed_trades",
        "n_long_trades",
        "n_short_trades",
        "turnover",
        "time_in_market",

        "lookback",
        "z_window",
        "entry_score",
        "exit_score",
        "mean_window_type",
        "signal_input",
        "vol_window",
        "fee_bps",
        "spread_filter",
        "spread_window",
        "spread_quantile",

        "spread_cost",
        "fee_cost",
        "avg_spread_bps",
        "median_spread_bps",
    ]

    existing_cols = [c for c in preferred_cols if c in summary.columns]
    other_cols = [c for c in summary.columns if c not in existing_cols]
    summary = summary[existing_cols + other_cols]

    out_path = Path(cfg["out_dir"]) / "summary.csv"
    summary.to_csv(out_path, index=False)

    print(summary.head(30))
    print(f"Saved summary to {out_path}")


    import matplotlib.pyplot as plt
    def log_heatmap(summary, symbol, metric="total_return_net"):
        df = summary[summary["symbol"] == symbol].copy()

        pivot = df.pivot_table(
            index="lookback",
            columns="entry_score",
            values=metric,
            aggfunc="mean"
        )

        plt.figure(figsize=(10, 6))
        plt.imshow(pivot, aspect="auto")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel("entry_score")
        plt.ylabel("lookback")
        plt.title(f"{symbol}: {metric} heatmap")
        plt.colorbar(label=metric)

        wandb.log({
            f"heatmap/{symbol}_{metric}_lookback_entry": wandb.Image(plt)
        })

        plt.close()

    if cfg.get("use_wandb", False):
        import wandb

        top200 = summary.sort_values(
            "total_return_net",
            ascending=False
        ).head(200)

        wandb.init(
            project=cfg.get("project", "crypto_micro_alpha"),
            name="grid_summary_top200",
            reinit=True,
        )

        wandb.log({
            "top200_table": wandb.Table(dataframe=top200)
        })

        wandb.finish()
    


if __name__ == "__main__":
    main()