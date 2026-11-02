"""List prediction gaps that still need a model call.

This script does not contact an API unless both OPENROUTER_API_KEY is set and
--execute is passed. The key has to be a rotated credential. The historical
runner sampled every other or every fourth image; the default here is stride 1
and only diagrams listed in the audit gap file.

Usage:
    python Codes/rerun_missing_cells.py
    python Codes/rerun_missing_cells.py --execute
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAPS = ROOT / "ERUnderstand" / "stats" / "audit" / "missing_predictions.csv"
SUMMARY = ROOT / "ERUnderstand" / "stats" / "audit" / "gaps_to_rerun.csv"

# These three families are the stride-2 subset. Completing them removes the
# comparability objection without rerunning models that already have full files.
PRIORITY_MODELS = {"Llama-3.2", "Gemma-3", "GLM-4.5"}
PRIORITY_TIERS = {
    "easy",
    "med",
    "hard",
    "Web",
    "Web-Silber",
    "Instructional",
    "Instructional-Silber",
}


def load_gaps() -> list:
    """Read the audit gap list written by reproduce_tables.py."""
    if not GAPS.exists():
        raise SystemExit(f"Missing {GAPS}. Run Codes/Evaluation/reproduce_tables.py first.")
    with GAPS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_summary(rows: list) -> None:
    """Aggregate missing diagrams by model and tier."""
    counts = Counter((row["model"], row["tier"], row["status"]) for row in rows)
    summary = [
        {"model": model, "tier": tier, "status": status, "n": count}
        for (model, tier, status), count in sorted(counts.items())
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "tier", "status", "n"])
        writer.writeheader()
        writer.writerows(summary)


def priority_rows(rows: list) -> list:
    """Keep the partial open-model tiers that the audit showed are stride-2 subsets."""
    return [
        row
        for row in rows
        if row["model"] in PRIORITY_MODELS and row["tier"] in PRIORITY_TIERS and row["status"] == "missing"
    ]


def main() -> None:
    """Print the gap summary and stop before any paid API call by default."""
    parser = argparse.ArgumentParser(description="Rerun only unresolved ERD cells")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Contact OpenRouter. Refuses to run unless OPENROUTER_API_KEY is set.",
    )
    args = parser.parse_args()
    rows = load_gaps()
    write_summary(rows)
    needed = priority_rows(rows)
    print(f"Priority missing predictions: {len(needed)}")
    print(f"Summary: {SUMMARY}")
    if not args.execute:
        print("Not calling the API. Re-run with --execute after rotating OPENROUTER_API_KEY.")
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set. Rotate the leaked key before any rerun.", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(
        "The gap list is ready, but image-to-tier path mapping for a live rerun "
        "still has to be confirmed against the rotated endpoint. No requests were sent."
    )


if __name__ == "__main__":
    main()
