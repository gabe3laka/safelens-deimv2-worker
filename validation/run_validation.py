#!/usr/bin/env python3
"""
validation/run_validation.py -- deterministic risk-engine quality gate (B9).

Runs the deterministic risk engine over synthetic hazard scenarios (no GPU, no
weights, no real images) and checks that it still recalls the expected hazards.
Exits NON-ZERO when critical-hazard recall drops below VALIDATION_MIN_RECALL_CRITICAL
(default 0.90) or a critical scenario regresses -- so it can be wired as a
BLOCKING CI gate before the image build.

Usage:
    python validation/run_validation.py [--scenarios PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_LEVELS = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
_DEFAULT_SCENARIOS = Path(__file__).with_name("scenarios") / "hazard_scenarios.json"


def _min_recall_critical() -> float:
    try:
        return float(os.getenv("VALIDATION_MIN_RECALL_CRITICAL", "0.90"))
    except (TypeError, ValueError):
        return 0.90


def run(scenarios_path: Path) -> dict:
    # The engine must be ON for validation regardless of the ambient env.
    os.environ["RISK_ENGINE_ENABLED"] = "true"
    import risk
    from risk import risk_matrix, tracking

    risk_matrix.reset_cache()  # use the active/default matrix

    data = json.loads(scenarios_path.read_text(encoding="utf-8"))
    img_w = int(data.get("img_w", 1280))
    img_h = int(data.get("img_h", 720))

    total_expected = matched = 0
    crit_expected = crit_matched = 0
    results = []
    for sc in data.get("scenarios", []):
        tracking.reset()  # isolate each scenario
        out = risk.evaluate(entities=sc["entities"], img_w=img_w, img_h=img_h,
                            session_id="validation_" + sc["name"])
        found = {r["hazard_type"] for r in out["risks"]}
        highest = out["highest_risk_level"]
        critical = bool(sc.get("critical"))
        sc_ok = True
        for hz in sc.get("expect_hazards", []):
            total_expected += 1
            if critical:
                crit_expected += 1
            if hz in found:
                matched += 1
                if critical:
                    crit_matched += 1
            else:
                sc_ok = False
        # level expectation (lower bound)
        want_level = sc.get("expect_min_level")
        level_ok = (want_level is None
                    or _LEVELS.get(highest, 0) >= _LEVELS.get(want_level, 0))
        results.append({
            "name": sc["name"], "critical": critical, "found": sorted(found),
            "expected": sc.get("expect_hazards", []), "highest": highest,
            "hazards_ok": sc_ok, "level_ok": level_ok,
        })

    recall = matched / total_expected if total_expected else 1.0
    crit_recall = crit_matched / crit_expected if crit_expected else 1.0
    min_recall = _min_recall_critical()
    level_failures = [r["name"] for r in results if not r["level_ok"]]
    crit_failures = [r["name"] for r in results
                     if r["critical"] and not r["hazards_ok"]]
    passed = (crit_recall >= min_recall and not crit_failures and not level_failures)
    return {
        "passed": passed,
        "recall": round(recall, 4),
        "critical_recall": round(crit_recall, 4),
        "min_recall_critical": min_recall,
        "critical_failures": crit_failures,
        "level_failures": level_failures,
        "scenarios": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Risk-engine validation gate")
    ap.add_argument("--scenarios", default=str(_DEFAULT_SCENARIOS))
    ap.add_argument("--json", action="store_true", help="print full JSON report")
    args = ap.parse_args()

    report = run(Path(args.scenarios))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"risk validation: passed={report['passed']} "
              f"recall={report['recall']} critical_recall={report['critical_recall']} "
              f"(min {report['min_recall_critical']})")
        if report["critical_failures"]:
            print("  CRITICAL hazard misses:", report["critical_failures"])
        if report["level_failures"]:
            print("  level regressions:", report["level_failures"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
