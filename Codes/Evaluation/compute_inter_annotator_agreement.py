#!/usr/bin/env python3
"""
Compute inter-annotator agreement between two JSON annotation directories.

Compares annotator A vs. annotator B per diagram using the same normalization and
error taxonomy as the main evaluation pipeline, then reports:
  - Per-category element agreement (F1 treating A as reference, B as prediction)
  - Diagram-level exact-match rate (zero structural diffs)
  - Macro-F1 agreement (mean of per-category F1 where GT count > 0)

Usage:
  python compute_inter_annotator_agreement.py \\
    --annotator-a /path/to/annotator_1 \\
    --annotator-b /path/to/annotator_2 \\
    --tier Web \\
    --output ERUnderstand/stats/inter_annotator_agreement.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "Codes" / "Evaluation"))

from compare import compare_entities, compare_relationships, normalize_json
from measure import FIELDNAMES, PATTERNS


def compare_json_pair(gt: Dict[str, Any], test: Dict[str, Any]) -> List[str]:
    """Return structural diff lines between two normalized JSON dicts."""
    diffs = compare_entities(gt.get("entities", {}), test.get("entities", {}))
    diffs += compare_relationships(gt.get("relationships", []), test.get("relationships", []))
    return diffs


def diffs_to_error_row(diagram: str, diffs: List[str]) -> Dict[str, int]:
    """Convert diff lines to error-count columns (same schema as measure.py)."""
    row: Dict[str, int] = {field: 0 for field in FIELDNAMES}
    row["Diagram"] = diagram
    for line in diffs:
        for pattern, col in PATTERNS:
            if pattern.search(line):
                row[col] = row.get(col, 0) + 1
                break
    row["Sum"] = sum(row.get(f, 0) for f in FIELDNAMES if f not in ("Diagram", "Sum"))
    return row


def compute_category_f1(gt_count: int, extra: int, missed: int, miscon: int, f: float = 0.5) -> float:
    """Category F1 with configurable misconnection weight."""
    if gt_count <= 0:
        return 1.0 if extra == 0 and missed == 0 and miscon == 0 else 0.0
    tp = max(0, gt_count - missed)
    fp = extra + (1 - f) * miscon
    fn = missed + f * miscon
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / gt_count if gt_count > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def count_gt_elements(data: Dict[str, Any]) -> Dict[str, int]:
    """Count GT element totals per evaluation category from one JSON."""
    entities = data.get("entities", {})
    relationships = data.get("relationships", [])

    n_ary = sum(1 for r in relationships if len(r.get("entities", []) or []) > 2)
    isa = sum(1 for r in relationships if "subclasses" in r or "superclass" in r)
    binary_rels = len(relationships) - n_ary - isa

    totals = {
        "entities": len(entities),
        "attributes": sum(len(e.get("attributes", []) or []) for e in entities.values()),
        "primary_keys": sum(len(e.get("primary_keys", []) or []) for e in entities.values()),
        "multivalued": sum(len(e.get("multivalued", []) or []) for e in entities.values()),
        "derived": sum(len(e.get("derived", []) or []) for e in entities.values()),
        "composite": sum(
            len((e.get("composite", {}) or {}).keys()) for e in entities.values()
        ),
        "weak_entities": sum(1 for e in entities.values() if e.get("weak")),
        "weak_relationships": sum(1 for r in relationships if r.get("weak")),
        "relationships": max(0, binary_rels),
        "cardinality": max(0, binary_rels),
        "IsA": isa,
        "n_ary": n_ary,
    }
    return totals


def aggregate_agreement(
    annotator_a: Path,
    annotator_b: Path,
    miscon_weight: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Compare paired JSON files across two annotation directories.

    Uses annotator A as reference and annotator B as comparison target.
    """
    per_diagram: List[Dict[str, Any]] = []
    category_error_sums = {k: {"extra": 0, "missed": 0, "miscon": 0, "gt": 0} for k in [
        "entities", "attributes", "primary_keys", "multivalued", "derived", "composite",
        "weak_entities", "weak_relationships", "relationships", "cardinality", "IsA", "n_ary",
    ]}

    error_map = {
        "entities": ("Ent-extra", "Ent-missed", None),
        "attributes": ("Attr-extra", "Attr-missed", "Attr-miscon"),
        "primary_keys": ("PrimKey-extra", "PrimKey-missed", None),
        "multivalued": (None, "multival-missed", None),
        "derived": (None, "derived-missed", None),
        "composite": ("comp-extra", "comp-missed", None),
        "relationships": ("Rel-extra", "Rel-missed", "Rel-miscon"),
        "cardinality": (None, None, "Cardinality"),
        "weak_entities": ("WeakEnt-extra", "WeakEnt-missed", None),
        "weak_relationships": ("WeakRel-extra", "WeakRel-missed", None),
        "IsA": ("IsA-extra", "IsA-missed", "IsA-miscon"),
        "n_ary": (None, None, "N-ary-missed"),
    }

    files_a = sorted(annotator_a.glob("*.json"))
    for path_a in files_a:
        path_b = annotator_b / path_a.name
        if not path_b.exists():
            continue

        json_a = normalize_json(json.loads(path_a.read_text(encoding="utf-8")))
        json_b = normalize_json(json.loads(path_b.read_text(encoding="utf-8")))
        diffs = compare_json_pair(json_a, json_b)
        err_row = diffs_to_error_row(path_a.stem, diffs)
        gt_counts = count_gt_elements(json_a)

        cat_f1s = []
        for cat, (ex_col, miss_col, misc_col) in error_map.items():
            gt = gt_counts.get("cardinality" if cat == "cardinality" else cat, 0)
            if cat == "cardinality":
                gt = gt_counts["relationships"]
            extra = err_row.get(ex_col, 0) if ex_col else 0
            missed = err_row.get(miss_col, 0) if miss_col else 0
            miscon = err_row.get(misc_col, 0) if misc_col else 0
            if gt > 0 or extra > 0 or missed > 0 or miscon > 0:
                category_error_sums[cat]["gt"] += gt
                category_error_sums[cat]["extra"] += extra
                category_error_sums[cat]["missed"] += missed
                category_error_sums[cat]["miscon"] += miscon
                cat_f1s.append(compute_category_f1(gt, extra, missed, miscon, miscon_weight))

        macro = sum(cat_f1s) / len(cat_f1s) if cat_f1s else 1.0
        per_diagram.append(
            {
                "Diagram": path_a.stem,
                "Exact_Match": int(len(diffs) == 0),
                "Diff_Lines": len(diffs),
                "Macro_F1_Agreement": round(macro, 3),
            }
        )

    summary = {
        "diagrams": len(per_diagram),
        "exact_match_rate": round(
            sum(d["Exact_Match"] for d in per_diagram) / max(1, len(per_diagram)), 3
        ),
        "mean_macro_f1_agreement": round(
            sum(d["Macro_F1_Agreement"] for d in per_diagram) / max(1, len(per_diagram)), 3
        ),
    }

    cat_f1_summary = {}
    for cat, sums in category_error_sums.items():
        if sums["gt"] > 0:
            cat_f1_summary[cat] = round(
                compute_category_f1(sums["gt"], sums["extra"], sums["missed"], sums["miscon"], miscon_weight),
                3,
            )
    summary["category_f1"] = cat_f1_summary

    return per_diagram, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-annotator agreement for ER JSON annotations")
    parser.add_argument("--annotator-a", required=True, help="Directory with annotator A JSON files")
    parser.add_argument("--annotator-b", required=True, help="Directory with annotator B JSON files")
    parser.add_argument("--tier", default="Web", help="Label for output (Web, Instructional, etc.)")
    parser.add_argument("--output", default="ERUnderstand/stats/inter_annotator_agreement.csv")
    parser.add_argument("--miscon-weight", type=float, default=0.5)
    args = parser.parse_args()

    per_diagram, summary = aggregate_agreement(
        Path(args.annotator_a), Path(args.annotator_b), args.miscon_weight
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Tier", "Diagram", "Exact_Match", "Diff_Lines", "Macro_F1_Agreement"],
        )
        writer.writeheader()
        for row in per_diagram:
            writer.writerow({"Tier": args.tier, **row})

    print(f"[OK] {args.tier}: {summary['diagrams']} diagrams")
    print(f"     Exact-match rate: {summary['exact_match_rate']}")
    print(f"     Mean macro-F1 agreement: {summary['mean_macro_f1_agreement']}")
    print(f"     Category F1: {summary['category_f1']}")
    print(f"     Wrote {out_path}")


if __name__ == "__main__":
    main()
