from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Config
from src.data.ingestion import FEATURE_COLUMNS, ingest_ticker, normalize_ticker
from src.data.preparation import prepare_sequences
from src.utils import initialize_dirs

FEATURE_STORE_DIR = Path(__file__).resolve().parents[2] / "feature_store"


def _cfg() -> Config:
    return Config()


def _resolve_tickers(tickers: list[str] | None, cfg: Config) -> list[str]:
    requested = tickers or [cfg.parent_ticker, *cfg.child_tickers]
    resolved: list[str] = []
    for ticker in requested:
        symbol = normalize_ticker(ticker)
        if symbol not in resolved:
            resolved.append(symbol)
    return resolved


def _range(frame: pd.DataFrame) -> dict[str, str]:
    dates = pd.to_datetime(frame["date"])
    return {"start": dates.min().strftime("%Y-%m-%d"), "end": dates.max().strftime("%Y-%m-%d")}


def build_dataset(
    tickers: list[str] | None = None,
    *,
    cfg: Config | None = None,
    downloader: Any | None = None,
) -> dict[str, Any]:
    """Ingest each ticker and persist validated features to the offline store."""
    config = cfg or _cfg()
    initialize_dirs(config.workdir.parent)
    summary: dict[str, Any] = {}
    for symbol in _resolve_tickers(tickers, config):
        frame = ingest_ticker(
            symbol,
            path=config.feature_path,
            start=config.start_date,
            period=None,
            context_length=config.context_len,
            prediction_length=config.pred_len,
            **({"downloader": downloader} if downloader is not None else {}),
        )
        summary[symbol] = {"rows": len(frame), **_range(frame)}
    return {"feature_path": str(config.feature_path), "tickers": summary}


def load_features(
    ticker: str,
    *,
    path: str | Path | None = None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Read one ticker back out of the offline store in training-ready shape."""
    config = cfg or _cfg()
    source = Path(path or config.feature_path)
    if not source.exists():
        raise FileNotFoundError(f"Feature store missing at {source}; run `sip-data build`")

    symbol = normalize_ticker(ticker)
    stored = pd.read_parquet(source)
    rows = stored.loc[stored["ticker"] == symbol]
    if rows.empty:
        raise KeyError(f"{symbol} absent from {source}; run `sip-data build --tickers {symbol}`")

    frame = rows.rename(columns={"event_timestamp": "date"}).loc[:, ["date", *FEATURE_COLUMNS]]
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None)
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].astype("float64")
    return (
        frame.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def dataset_report(*, path: str | Path | None = None, cfg: Config | None = None) -> dict[str, Any]:
    """Describe stored coverage and the chronological split each ticker would produce."""
    config = cfg or _cfg()
    source = Path(path or config.feature_path)
    if not source.exists():
        return {"feature_path": str(source), "exists": False, "tickers": {}}

    stored = pd.read_parquet(source)
    report: dict[str, Any] = {}
    for symbol in sorted(stored["ticker"].unique()):
        frame = load_features(symbol, path=source, cfg=config)
        entry: dict[str, Any] = {
            "rows": len(frame),
            **_range(frame),
            "missing_values": int(frame[FEATURE_COLUMNS].isna().to_numpy().sum()),
        }
        try:
            prepared = prepare_sequences(
                frame,
                context_length=config.context_len,
                prediction_length=config.pred_len,
                train_fraction=config.train_ratio,
                val_fraction=config.validation_ratio,
                feature_columns=tuple(config.features),
                target_column="Close",
            )
            entry["splits"] = {
                "train": len(prepared.train),
                "val": len(prepared.val),
                "test": len(prepared.test),
            }
        except ValueError as exc:
            entry["splits_error"] = str(exc)
        report[str(symbol)] = entry
    return {"feature_path": str(source), "exists": True, "tickers": report}


def materialize_feature_store(*, cfg: Config | None = None) -> dict[str, Any]:
    """Run Feast apply and incremental materialization, reporting failures explicitly."""
    config = cfg or _cfg()
    if not Path(config.feature_path).exists():
        raise FileNotFoundError(f"Nothing to materialize; {config.feature_path} is absent")

    commands = [
        ["feast", "apply"],
        ["feast", "materialize-incremental", datetime.now(UTC).isoformat()],
    ]
    steps: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=FEATURE_STORE_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        steps.append(
            {
                "command": " ".join(command),
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip()[-800:],
            }
        )
        if completed.returncode != 0:
            break
    return {"ok": all(step["returncode"] == 0 for step in steps), "steps": steps}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and inspect the offline feature store")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Ingest tickers into the feature store")
    build.add_argument("--tickers", nargs="+", default=None)

    subparsers.add_parser("inspect", help="Report stored coverage and split sizes")
    subparsers.add_parser("materialize", help="Push stored features to the online store")

    args = parser.parse_args(argv)
    if args.command == "build":
        payload = build_dataset(args.tickers)
    elif args.command == "inspect":
        payload = dataset_report()
    else:
        payload = materialize_feature_store()
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
