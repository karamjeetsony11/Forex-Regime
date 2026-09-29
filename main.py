"""Terminal control center for the standalone FX market regime detector.

Run ``python main.py`` for the interactive menu, or use direct commands:
    python main.py fetch --symbol EURUSD
    python main.py features --symbol EURUSD
    python main.py train
    python main.py predict --symbol EURUSD
    python main.py run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.data.forex_loader import DEFAULT_SYMBOLS, ForexDataLoader

ROOT = Path(__file__).resolve().parent


def run_script(script: str, *args: str) -> None:
    """Run one of the existing pipeline stages with the active interpreter."""
    subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, check=True)


def fetch_data(symbol: str | None, limit: int = 500) -> None:
    symbols = [symbol] if symbol else DEFAULT_SYMBOLS
    loader = ForexDataLoader(symbols=symbols)
    results = loader.fetch_all(interval="1h", ohlcv_limit=limit)
    for pair, data in results.items():
        print(f"Saved {len(data['ohlcv'])} hourly candles for {pair}.")


def run_pipeline(symbol: str | None) -> None:
    print("\n[1/4] Downloading hourly FX candles …")
    fetch_data(symbol)
    print("\n[2/4] Building features …")
    feature_args = ["--symbol", symbol] if symbol else []
    run_script("run_features.py", *feature_args)
    print("\n[3/4] Training the FX regime model …")
    run_script("run_training.py")
    print("\n[4/4] Predicting the current regime …")
    run_script("predict.py", "--symbol", symbol or DEFAULT_SYMBOLS[0])


def show_status() -> None:
    print("\nFX Market Regime Detector — local project status")
    print(f"Supported pairs: {', '.join(DEFAULT_SYMBOLS)}")
    print(f"Raw candle files: {len(list((ROOT / 'data' / 'raw').glob('*_ohlcv_*.csv'))) if (ROOT / 'data' / 'raw').exists() else 0}")
    print(f"Feature dataset: {'ready' if (ROOT / 'data' / 'features' / 'features.parquet').exists() else 'not built'}")
    required = [ROOT / "models" / name for name in ("scaler.pkl", "pca.pkl", "regime_model.pkl")]
    print(f"Trained model: {'ready' if all(path.exists() for path in required) else 'not trained'}")


def show_report() -> None:
    path = ROOT / "reports" / "regime_report.txt"
    if not path.exists():
        print("No training report yet. Choose Train model first.")
        return
    print(path.read_text(encoding="utf-8"))


def prompt_pair() -> str:
    while True:
        value = input(f"FX pair [{DEFAULT_SYMBOLS[0]}]: ").strip().upper() or DEFAULT_SYMBOLS[0]
        try:
            pair = ForexDataLoader.normalize_symbol(value)
        except ValueError as exc:
            print(exc)
            continue
        if pair not in DEFAULT_SYMBOLS:
            print(f"Choose one of: {', '.join(DEFAULT_SYMBOLS)}")
            continue
        return pair


def interactive_menu() -> None:
    actions = {
        "1": ("Download FX candles", lambda: fetch_data(None)),
        "2": ("Download candles for one pair", lambda: fetch_data(prompt_pair())),
        "3": ("Build features", lambda: run_script("run_features.py")),
        "4": ("Train / refresh model", lambda: run_script("run_training.py")),
        "5": ("Predict current market regime", lambda: run_script("predict.py", "--symbol", prompt_pair())),
        "6": ("Run full pipeline", lambda: run_pipeline(None)),
        "7": ("Show project status", show_status),
        "8": ("Show latest training report", show_report),
    }
    while True:
        print("\n╭──────────────────────────────────────────╮")
        print("│       FX MARKET REGIME DETECTOR           │")
        print("├──────────────────────────────────────────┤")
        for key, (label, _) in actions.items():
            print(f"│  {key}. {label:<37}│")
        print("│  0. Exit                                  │")
        print("╰──────────────────────────────────────────╯")
        choice = input("Select an option: ").strip()
        if choice == "0":
            print("Done.")
            return
        action = actions.get(choice)
        if not action:
            print("Please enter an option from 0 to 8.")
            continue
        try:
            action[1]()
        except Exception as exc:
            print(f"\nThat step failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone terminal toolkit for FX market regime analysis.",
        epilog="Run without a command to open the interactive terminal menu.",
    )
    commands = parser.add_subparsers(dest="command")

    fetch = commands.add_parser("fetch", help="Download hourly FX candles")
    fetch.add_argument("--symbol", choices=DEFAULT_SYMBOLS, help="Pair to download; defaults to all supported pairs")
    fetch.add_argument("--limit", type=int, default=500, help="Candles per pair (default: 500)")

    features = commands.add_parser("features", help="Build features from downloaded candles")
    features.add_argument("--symbol", choices=DEFAULT_SYMBOLS)
    features.add_argument("--show-sample", action="store_true")

    train = commands.add_parser("train", help="Train or refresh the FX regime model")
    train.add_argument("--n-clusters", type=int, default=6)
    train.add_argument("--pca-variance", type=float, default=0.95)

    predict = commands.add_parser("predict", help="Predict the current regime for one pair")
    predict.add_argument("--symbol", choices=DEFAULT_SYMBOLS, default=DEFAULT_SYMBOLS[0])
    predict.add_argument("--json", action="store_true")

    pipeline = commands.add_parser("run", help="Run download, features, training, and prediction")
    pipeline.add_argument("--symbol", choices=DEFAULT_SYMBOLS, help="Run for one pair; defaults to all pairs")
    commands.add_parser("status", help="Show local data and model readiness")
    commands.add_parser("report", help="Print the most recent training report")
    commands.add_parser("menu", help="Open the interactive terminal menu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in (None, "menu"):
        interactive_menu()
    elif args.command == "fetch":
        fetch_data(args.symbol, args.limit)
    elif args.command == "features":
        feature_args = ["--symbol", args.symbol] if args.symbol else []
        if args.show_sample:
            feature_args.append("--show-sample")
        run_script("run_features.py", *feature_args)
    elif args.command == "train":
        run_script("run_training.py", "--n-clusters", str(args.n_clusters), "--pca-variance", str(args.pca_variance))
    elif args.command == "predict":
        predict_args = ["--symbol", args.symbol]
        if args.json:
            predict_args.append("--json")
        run_script("predict.py", *predict_args)
    elif args.command == "run":
        run_pipeline(args.symbol)
    elif args.command == "status":
        show_status()
    elif args.command == "report":
        show_report()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
