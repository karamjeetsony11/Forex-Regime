# FX Market Regime Detector

A standalone, terminal-first Python project for exploring market regimes in major foreign-exchange pairs. It downloads hourly FX candles from Yahoo Finance, engineers price features, trains an unsupervised model, and prints predictions and reports in the terminal. It has no dashboard or Node.js setup.

Supported pairs: `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`, `USDCAD`, `USDCHF`, `NZDUSD`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Interactive terminal menu

```bash
python main.py
```

The menu lets you download candles, build features, train the model, run a prediction, run the full pipeline, inspect local project status, and print the latest training report.

## Direct commands

```bash
# Download hourly candles for every supported pair, or just one pair
python main.py fetch
python main.py fetch --symbol EURUSD

# Generate features, train, and predict
python main.py features
python main.py train
python main.py predict --symbol EURUSD

# Run all steps for all pairs, or for one pair
python main.py run
python main.py run --symbol GBPUSD

# Inspect current local state and the last report
python main.py status
python main.py report
```

The underlying stage scripts remain available if you want their detailed logs:

```bash
python run_features.py --symbol EURUSD --show-sample
python run_training.py --n-clusters 6 --pca-variance 0.95
python predict.py --symbol EURUSD --json
```

The full pipeline saves candles under `data/raw/`, features under `data/features/`, model artifacts under `models/`, and a human-readable training report under `reports/`. Running training replaces the existing scaler, PCA, and regime model with artifacts trained from the current FX dataset. The inference engine rejects artifacts that are not marked FX-trained.

## Data and model notes

The loader uses Yahoo Finance hourly candles and does not need an API key. FX has no centralized exchange tape or order book, so volume compatibility features use neutral placeholder values; they are not real traded volume. The model describes statistical clusters and is for analysis, not a trading signal.

## Optional local API

The terminal workflow does not need the API. If you want to use the REST interface separately, install `requirements-deploy.txt` and run:

```bash
python run_api.py
```

It serves the same FX model from FastAPI. There is no frontend included or required.
