"""Run the parent refresh and child experiment matrix."""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.config import Config
from src.pipelines.training_pipeline import train_child, train_parent


def _child_matrix(tickers: list[str], source: str) -> list[dict[str, Any]]:
    cfg = Config()
    experiments = [
        {"label": "full_20", "transfer_strategy": "full", "epochs": 20, "child_improvement": 0.01},
        {"label": "freeze_30", "transfer_strategy": "freeze", "epochs": 30, "child_improvement": 0.01},
    ]
    results: list[dict[str, Any]] = []
    for ticker in tickers:
        for experiment in experiments:
            summary = train_child(
                ticker,
                epochs=experiment["epochs"],
                transfer_strategy=experiment["transfer_strategy"],
                child_improvement=experiment["child_improvement"],
                source=source,
            )
            results.append(
                {
                    "ticker": ticker,
                    "experiment": experiment["label"],
                    **summary,
                }
            )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run parent refresh and child experiments")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=Config().child_tickers,
        help="Child tickers to experiment on",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "feature-store", "yfinance"],
        default="feature-store",
    )
    parser.add_argument("--parent-epochs", type=int, default=30)
    parser.add_argument("--skip-parent", action="store_true")
    parser.add_argument("--skip-children", action="store_true")
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {"parent": None, "children": []}
    if not args.skip_parent:
        payload["parent"] = train_parent(epochs=args.parent_epochs, source=args.source)
    if not args.skip_children:
        payload["children"] = _child_matrix(args.tickers, args.source)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
