from pathlib import Path
import uuid
import numpy as np
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ResearchLogger:
    def __init__(self, out_dir, project="crypto_micro_alpha", use_wandb=False):
        self.out_dir = Path(out_dir)
        self.project = project
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        for subdir in ["experiments", "equity", "daily", "trades"]:
            (self.out_dir / subdir).mkdir(parents=True, exist_ok=True)

    def log_run(self, config, metrics, equity_df=None, daily_df=None, trades_df=None):
        run_id = str(uuid.uuid4())[:8]

        row = {"run_id": run_id, **config, **metrics}
        pd.DataFrame([row]).to_parquet(
            self.out_dir / "experiments" / f"{run_id}.parquet",
            index=False,
        )

        if equity_df is not None:
            equity_df = equity_df.copy()
            equity_df["run_id"] = run_id
            equity_df.to_parquet(
                self.out_dir / "equity" / f"{run_id}.parquet",
                index=False,
            )

        if daily_df is not None:
            daily_df = daily_df.copy()
            daily_df["run_id"] = run_id
            daily_df.to_parquet(
                self.out_dir / "daily" / f"{run_id}.parquet",
                index=False,
            )

        if trades_df is not None and len(trades_df) > 0:
            trades_df = trades_df.copy()
            trades_df["run_id"] = run_id
            trades_df.to_parquet(
                self.out_dir / "trades" / f"{run_id}.parquet",
                index=False,
            )

        if self.use_wandb:
            wandb.init(
                project=self.project,
                config=config,
                name=run_id,
                reinit=True,
            )
            wandb.log(metrics)

            if equity_df is not None and len(equity_df) > 0:
                wandb.log({
                    "equity_net_final": metrics.get("net_pnl", np.nan),
                })

            wandb.finish()

        return run_id