import argparse
from pathlib import Path
from itertools import product

import pandas as pd
from tqdm import tqdm

from src.data import load_data
from src.backtest import run_mr_backtest
from src.metrics import make_metrics, make_daily, make_equity, make_trades, make_trade_metrics
from src.logger import ResearchLogger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="research_logs")

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["act", "bsb", "siren", "tac"],
    )

    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--project", type=str, default="crypto_micro_alpha")
    parser.add_argument("--use-wandb", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    dfs = load_data(args.data_dir, args.symbols)

    logger = ResearchLogger(
        out_dir=args.out_dir,
        project=args.project,
        use_wandb=args.use_wandb,
    )

    '''
    lookbacks = [300, 600, 1800, 2400, 3600, 7200]
    entries = [1.5, 2.0, 2.5, 3.0, 3.25, 3.5]
    exits = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]
    spread_filters = ["none", "low", "high"]
    spread_quantiles = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]'''
    lookbacks = [2400, 7200]
    entries = [3.0, 3.5]
    exits = [0.1, 0.5]
    spread_filters = ["none"]
    spread_quantiles = [0.3]

    all_runs = []

    grid = list(product(args.symbols, lookbacks, entries, exits, spread_filters, spread_quantiles))

    for symbol, lookback, entry_score, exit_score, spread_filter, spread_quantile in tqdm(grid):
        if exit_score >= entry_score:
            continue

        z_window = max(1800, lookback * 10)
        spread_window = z_window

        config = {
            "strategy": "mean_reversion_bid_ask",
            "symbol": symbol,
            "lookback": lookback,
            "z_window": z_window,
            "entry_score": entry_score,
            "exit_score": exit_score,
            "fee_bps": args.fee_bps,
            "spread_filter": spread_filter,
            "spread_window": spread_window,
            "spread_quantile": spread_quantile,
        }

        bt = run_mr_backtest(
            dfs[symbol],
            lookback=lookback,
            z_window=z_window,
            entry_score=entry_score,
            exit_score=exit_score,
            fee_bps=args.fee_bps,
            spread_filter=spread_filter,
            spread_window=spread_window,
            spread_quantile=spread_quantile,
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

    out_path = Path(args.out_dir) / "summary.csv"
    summary.to_csv(out_path, index=False)
    summary = pd.DataFrame(all_runs).sort_values("log_net_pnl", ascending=False)
    out_path = Path(args.out_dir) / "summary.csv"
    summary.to_csv(out_path, index=False)

    print(summary.head(30))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()