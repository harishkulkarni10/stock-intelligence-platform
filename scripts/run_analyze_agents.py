"""Run analyze path pieces from the CLI: forecast, agent1, agent2 (no UI)."""

from __future__ import annotations

import argparse
import json

from src.agents.graph import analyze_stock
from src.agents.nodes import market_expert_node, performance_analyst_node
from src.agents.tools import format_forecast_for_prompt, get_forecast, get_news
from src.monitoring.agent_eval import evaluate_all_agent_fixtures


def _print_block(title: str, body: str) -> None:
    print("=" * 72)
    print(title)
    print("-" * 72)
    print(body)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run deterministic agent1+agent2 fixture evals (no Ollama / live news)",
    )
    parser.add_argument(
        "--full-analyze",
        action="store_true",
        help="Run analyze_stock (forecast → agent1 → agent2)",
    )
    args = parser.parse_args(argv)
    ticker = args.ticker.upper()

    if args.eval_only:
        report = evaluate_all_agent_fixtures()
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["ok"] else 1)

    if args.full_analyze:
        result = analyze_stock(ticker)
        print(json.dumps(result, indent=2, default=str))
        raise SystemExit(0 if result.get("status") == "ok" else 1)

    forecast = get_forecast(ticker)
    _print_block(
        "FORECAST (code, not LLM)",
        json.dumps(
            {
                "status": forecast.get("status"),
                "model_source": forecast.get("model_source"),
                "model_version": forecast.get("model_version"),
                "last_close": forecast.get("last_close"),
                "last_date": forecast.get("last_date"),
                "predictions": forecast.get("predictions"),
            },
            indent=2,
            default=str,
        ),
    )
    if forecast.get("status") != "ok":
        raise SystemExit(1)

    forecast_text = format_forecast_for_prompt(forecast)
    agent1 = performance_analyst_node(
        {"ticker": ticker, "forecast": forecast, "forecast_text": forecast_text}
    )
    _print_block(
        "AGENT 1 · Performance Analyst",
        agent1["performance_analysis"]
        + "\n\n"
        + json.dumps(
            {
                "performance_trend": agent1.get("performance_trend"),
                "performance_guardrail_ok": agent1.get("performance_guardrail_ok"),
                "performance_repaired": agent1.get("performance_repaired"),
            },
            indent=2,
        ),
    )

    news = get_news(ticker)
    _print_block(
        "NEWS TOOL (code, not LLM)",
        json.dumps(
            {
                "status": news.get("status"),
                "provider": news.get("provider"),
                "articles": news.get("articles"),
                "error": news.get("error"),
            },
            indent=2,
            default=str,
        ),
    )
    agent2 = market_expert_node({"ticker": ticker})
    _print_block(
        "AGENT 2 · Market Expert",
        agent2["news_summary"]
        + "\n\n"
        + json.dumps(
            {
                "news_sentiment": agent2.get("news_sentiment"),
                "news_guardrail_ok": agent2.get("news_guardrail_ok"),
                "news_repaired": agent2.get("news_repaired"),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
