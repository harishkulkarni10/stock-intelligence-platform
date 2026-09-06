"""Run only the Performance Analyst (agent 1) with guardrails — no UI / full graph."""

from __future__ import annotations

import argparse
import json

from src.agents.nodes import performance_analyst_node
from src.agents.tools import format_forecast_for_prompt, get_forecast
from src.monitoring.agent_eval import evaluate_performance_fixtures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run deterministic fixture evals (no Ollama)",
    )
    args = parser.parse_args(argv)

    if args.eval_only:
        report = evaluate_performance_fixtures()
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ok"] else 1)

    forecast = get_forecast(args.ticker)
    print("forecast status:", forecast.get("status"))
    print("model_source:", forecast.get("model_source"))
    if forecast.get("status") != "ok":
        raise SystemExit(json.dumps(forecast, indent=2))

    forecast_text = format_forecast_for_prompt(forecast)
    print("--- forecast_text ---")
    print(forecast_text)

    out = performance_analyst_node(
        {
            "ticker": args.ticker.upper(),
            "forecast": forecast,
            "forecast_text": forecast_text,
        }
    )
    print("--- performance_analysis ---")
    print(out["performance_analysis"])
    print("--- guardrails ---")
    print(
        json.dumps(
            {
                "performance_trend": out.get("performance_trend"),
                "performance_guardrail_ok": out.get("performance_guardrail_ok"),
                "performance_repaired": out.get("performance_repaired"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
