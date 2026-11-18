#!/usr/bin/env python3
"""Phase 0 offline baseline — deterministic Research + live-news unit paths (no live APIs).

Print JSON to stdout; copy results into phases/agentic-research/phase-00-baseline/.
"""

from __future__ import annotations

import json
import sys
import time

from src.monitoring.agent_eval import evaluate_research_fixtures


def main() -> int:
    t0 = time.perf_counter()
    research = evaluate_research_fixtures()
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    payload = {
        "phase": "0",
        "kind": "offline_baseline",
        "elapsed_ms": elapsed_ms,
        "research_ok": research.get("ok"),
        "research": research,
    }
    # JSON only on stdout (logs may go to stderr).
    sys.stdout.write(json.dumps(payload, indent=2, default=str))
    sys.stdout.write("\n")
    return 0 if research.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())