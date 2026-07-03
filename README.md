# crypto_micro_alpha

Microstructure alpha research for illiquid crypto futures using L1 bid/ask data.

## Local run

```bash
pip install -r requirements.txt

python -m src.experiments.grid_search \
  --data-dir ./data \
  --out-dir ./research_logs \
  --symbols act bsb siren tac \
  --fee-bps 10