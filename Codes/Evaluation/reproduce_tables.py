"""Regenerate coverage, Macro-F1, category F1, and sampling audits from artifacts.

Run from the repository root:

    python Codes/Evaluation/reproduce_tables.py

The script does not call a model API. It reads ground-truth JSON, archived
predictions, and the existing score CSVs, then writes tables under
ERUnderstand/stats/audit/.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare import compare_entities, compare_relationships, normalize_json  # noqa: E402
from measure import PATTERNS  # noqa: E402
from metric_policy import (  # noqa: E402
    CONDITIONAL_STATUSES,
    LENIENT_STATUSES,
    POLICY_CONDITIONAL,
    POLICY_LENIENT,
    POLICY_STRICT,
)

OUT_DIR = ROOT / "ERUnderstand" / "stats" / "audit"
AGGREGATE_CSV = ROOT / "ERUnderstand" / "stats" / "aggregated_sums.csv"

MODELS = [
    "Claude-4.6-sonnet",
    "GPT-5.2",
    "Gemini-3-pro",
    "Grok-2-vision",
    "Qwen-3.6",
    "Llama-3.2",
    "Gemma-3",
    "GLM-4.5",
]

# Paper tiers used for Table 2. Ground truth is shared across notation variants.
TIERS = [
    {
        "tier": "easy",
        "gt": "ERUnderstand/Generated/Graphviz-JSON/easy",
        "pred": "Experiments/Graphviz/Graphviz-easy",
        "images": "ERUnderstand/Generated/Graphviz/Graphviz-easy",
        "stats": "ERUnderstand/stats/sorted_stats_easy.csv",
        "group": "synthetic",
    },
    {
        "tier": "med",
        "gt": "ERUnderstand/Generated/Graphviz-JSON/med",
        "pred": "Experiments/Graphviz/Graphviz-med",
        "images": "ERUnderstand/Generated/Graphviz/Graphviz-med",
        "stats": "ERUnderstand/stats/sorted_stats_med.csv",
        "group": "synthetic",
    },
    {
        "tier": "hard",
        "gt": "ERUnderstand/Generated/Graphviz-JSON/hard",
        "pred": "Experiments/Graphviz/Graphviz-hard",
        "images": "ERUnderstand/Generated/Graphviz/Graphviz-hard",
        "stats": "ERUnderstand/stats/sorted_stats_hard.csv",
        "group": "synthetic",
    },
    {
        "tier": "Web",
        "gt": "ERUnderstand/Real-world/JSON/Web-JSON",
        "pred": "Experiments/Real-world/Web",
        "images": "ERUnderstand/Real-world/Diagrams/Web",
        "stats": "ERUnderstand/stats/sorted_stats_Web.csv",
        "group": "curated",
    },
    {
        "tier": "Web-Silber",
        "gt": "ERUnderstand/Real-world/JSON/Web-JSON",
        "pred": "Experiments/Real-world/Web-Silber",
        "images": "ERUnderstand/Real-world/Diagrams/Web-Silber",
        "stats": "ERUnderstand/stats/sorted_stats_Web.csv",
        "group": "curated",
    },
    {
        "tier": "Instructional",
        "gt": "ERUnderstand/Real-world/JSON/Instructional-JSON",
        "pred": "Experiments/Real-world/Instructional",
        "images": "ERUnderstand/Real-world/Diagrams/Instructional",
        "stats": "ERUnderstand/stats/sorted_stats_Instructional.csv",
        "group": "curated",
    },
    {
        "tier": "Instructional-Silber",
        "gt": "ERUnderstand/Real-world/JSON/Instructional-JSON",
        "pred": "Experiments/Real-world/Instructional-Silber",
        "images": "ERUnderstand/Real-world/Diagrams/Instructional-Silber",
        "stats": "ERUnderstand/stats/sorted_stats_Instructional.csv",
        "group": "curated",
    },
    {
        "tier": "Schema",
        "gt": "ERUnderstand/Real-world/JSON/Schema-JSON",
        "pred": "Experiments/Real-world/Schema",
        "images": "ERUnderstand/Real-world/Diagrams/Schema-Graphviz",
        "stats": "ERUnderstand/stats/sorted_stats_Schema.csv",
        "group": "schema",
    },
    {
        "tier": "extreme",
        "gt": "ERUnderstand/Generated/Graphviz-JSON/extreme",
        "pred": "Experiments/Graphviz/Graphviz-extreme",
        "images": "ERUnderstand/Generated/Graphviz/Graphviz-extreme",
        "stats": "ERUnderstand/stats/sorted_stats_extreme.csv",
        "group": "stress",
    },
    {
        "tier": "IsA",
        "gt": "ERUnderstand/Generated/Graphviz-JSON/IsA",
        "pred": "Experiments/Graphviz/Graphviz-IsA",
        "images": "ERUnderstand/Generated/Graphviz/Graphviz-IsA",
        "stats": "ERUnderstand/stats/sorted_stats_IsA.csv",
        "group": "stress",
    },
]

CATEGORIES = [
    ("Entities", "entities"),
    ("RegAttr", "attributes"),
    ("Multivalued", "multivalued"),
    ("Derived", "derived"),
    ("Composite", "composite"),
    ("PrimaryKeys", "primary_keys"),
    ("Relationships", "relationships"),
    ("Cardinality", "cardinality"),
    ("WeakEntities", "weak_entities"),
    ("WeakRelationships", "weak_relationships"),
    ("IsA", "IsA"),
    ("Nary", "n_ary"),
]

ERROR_FIELDS = {
    "Entities": ("Ent-extra", "Ent-missed", None),
    "RegAttr": ("Attr-extra", "Attr-missed", "Attr-miscon"),
    "Multivalued": (None, "multival-missed", None),
    "Derived": (None, "derived-missed", None),
    "Composite": ("comp-extra", "comp-missed", None),
    "PrimaryKeys": ("PrimKey-extra", "PrimKey-missed", None),
    "Relationships": ("Rel-extra", "Rel-missed", "Rel-miscon"),
    "Cardinality": (None, "Cardinality", None),
    "WeakEntities": ("WeakEnt-extra", "WeakEnt-missed", None),
    "WeakRelationships": ("WeakRel-extra", "WeakRel-missed", None),
    "IsA": ("IsA-extra", "IsA-missed", "IsA-miscon"),
    "Nary": (None, "N-ary-missed", None),
}

REFUSAL_RE = re.compile(
    r"\b(i cannot|i can't|i am unable|i'm unable|sorry|as an ai|refuse[sd]?|not able to)\b",
    re.IGNORECASE,
)
FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

# Base rows in aggregated_sums.csv. Variant rows are instances, not new schemas.
UNIQUE_ROWS = {
    "Web",
    "Instructional",
    "Schema",
    "easy",
    "medium",
    "hard",
    "extreme",
    "IsA",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of row dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write rows to a CSV, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def count_diagram(diagram: Dict[str, Any]) -> Dict[str, int]:
    """Count evaluation categories the same way the stats generator does.

    Regular-attribute totals include composite, multivalued, and derived names
    because that is the column F1 scoring reads as GT_RegAttr. Cardinality uses
    the same relationship denominator as the relationship category.
    """
    stats = {
        "entities": 0,
        "relationships": 0,
        "attributes": 0,
        "weak_entities": 0,
        "weak_relationships": 0,
        "n_ary": 0,
        "unary": 0,
        "IsA": 0,
        "derived": 0,
        "composite": 0,
        "multivalued": 0,
        "primary_keys": 0,
    }
    entities = diagram.get("entities") or {}
    stats["entities"] = len(entities)
    for entity_info in entities.values():
        if not isinstance(entity_info, dict):
            continue
        stats["attributes"] += len(entity_info.get("attributes") or [])
        composite = entity_info.get("composite") or {}
        if isinstance(composite, dict):
            stats["composite"] += len(composite)
            stats["attributes"] += len(composite)
            for sub_attrs in composite.values():
                stats["attributes"] += len(sub_attrs or [])
        stats["multivalued"] += len(entity_info.get("multivalued") or [])
        stats["attributes"] += len(entity_info.get("multivalued") or [])
        stats["derived"] += len(entity_info.get("derived") or [])
        stats["attributes"] += len(entity_info.get("derived") or [])
        stats["primary_keys"] += len(entity_info.get("primary_keys") or [])
        if entity_info.get("weak"):
            stats["weak_entities"] += 1

    for rel in diagram.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        involved = rel.get("entities") or []
        stats["attributes"] += len(rel.get("attributes") or [])
        if len(involved) == 2 and involved[0] == involved[1]:
            stats["unary"] += 1
        elif len(involved) > 2 and "isa" not in str(rel.get("name", "")).lower():
            stats["n_ary"] += 1
        if "rel_name" in rel:
            stats["IsA"] += 1
        else:
            stats["relationships"] += 1
        if rel.get("weak"):
            stats["weak_relationships"] += 1
    stats["cardinality"] = stats["relationships"]
    return stats


def compute_f1(gt: int, extra: int, missed: int, miscon: int) -> float:
    """Fractional F1 with misconnections split equally between FP and FN."""
    if gt <= 0:
        return 0.0
    true_positive = max(0, gt - missed)
    false_positive = extra + 0.5 * miscon
    false_negative = missed + 0.5 * miscon
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / gt if gt else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def count_diff_errors(diffs: Iterable[str]) -> Dict[str, int]:
    """Map comparison lines onto the same error columns as measure.py."""
    counts = {name: 0 for names in ERROR_FIELDS.values() for name in names if name}
    for line in diffs:
        for pattern, field in PATTERNS:
            if pattern.search(line):
                counts[field] = counts.get(field, 0) + 1
                break
    return counts


def score_prediction(gt_json: Dict[str, Any], pred_json: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Score one diagram. A missing prediction is an empty schema."""
    gt_counts = count_diagram(gt_json)
    prediction = pred_json if pred_json is not None else {"entities": {}, "relationships": []}
    if not isinstance(prediction, dict):
        prediction = {"entities": {}, "relationships": []}
    gt_norm = normalize_json(gt_json)
    pred_norm = normalize_json(prediction)
    diffs = compare_entities(gt_norm.get("entities", {}), pred_norm.get("entities", {}))
    diffs += compare_relationships(gt_norm.get("relationships", []), pred_norm.get("relationships", []))
    errors = count_diff_errors(diffs)
    scored: Dict[str, float] = {}
    supported: List[float] = []
    for label, key in CATEGORIES:
        extra_name, miss_name, miscon_name = ERROR_FIELDS[label]
        f1 = compute_f1(
            gt_counts[key],
            errors.get(extra_name or "", 0),
            errors.get(miss_name or "", 0),
            errors.get(miscon_name or "", 0),
        )
        scored[label] = f1
        scored[f"gt_{label}"] = gt_counts[key]
        if gt_counts[key] > 0:
            supported.append(f1)
    scored["macro"] = sum(supported) / len(supported) if supported else 0.0
    return scored


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull one JSON object out of a fence or surrounding prose."""
    match = FENCE_RE.search(text)
    candidate = match.group(1) if match else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidate = text[start : end + 1]
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def classify_file(path: Path) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, int]]:
    """Classify an archived prediction without calling a model."""
    flags = {"fenced": 0, "preamble": 0, "refusal": 0, "truncated": 0}
    if not path.exists():
        return "missing", None, flags
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = text.strip()
    flags["fenced"] = int("```" in text)
    flags["preamble"] = int(bool(stripped) and not stripped.startswith("{") and not stripped.startswith("["))
    flags["refusal"] = int(bool(REFUSAL_RE.search(text)))
    flags["truncated"] = int(bool(stripped) and not stripped.endswith("}") and not stripped.endswith("]"))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        extracted = extract_json_object(text)
        if extracted is not None:
            return "lenient_extracted", extracted, flags
        if flags["refusal"]:
            return "refusal", None, flags
        if flags["truncated"]:
            return "truncated", None, flags
        return "invalid_json", None, flags
    if not isinstance(parsed, dict):
        return "invalid_json", None, flags
    entities = parsed.get("entities") or {}
    relationships = parsed.get("relationships") or []
    if not entities and not relationships:
        return "empty_schema", parsed, flags
    return "valid_json", parsed, flags


def load_tier_ground_truth(spec: Dict[str, str]) -> List[Dict[str, Any]]:
    """Load every ground-truth JSON for a tier and its archived stats row."""
    gt_dir = ROOT / spec["gt"]
    stats_rows = {}
    stats_path = ROOT / spec["stats"]
    if stats_path.exists():
        for row in read_csv(stats_path):
            if row.get("Diagram Number") != "Sum":
                stats_rows[row["Diagram Number"]] = row
    diagrams = []
    for path in sorted(gt_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        fresh = count_diagram(payload)
        archived = stats_rows.get(path.stem)
        diagrams.append(
            {
                "stem": path.stem,
                "json": payload,
                "fresh": fresh,
                "stats_entities": int(archived["entities"]) if archived and archived.get("entities") else None,
            }
        )
    return diagrams


def image_stems(spec: Dict[str, str]) -> List[str]:
    """Return sorted image stems when a diagram folder exists."""
    image_dir = ROOT / spec["images"]
    if not image_dir.exists():
        return []
    stems = []
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            stems.append(path.stem)
    return sorted(stems)


def describe_sampling(image_names: List[str], prediction_stems: Iterable[str]) -> str:
    """Say whether predictions are a full set or a fixed stride of the sorted images."""
    names = image_names
    predicted = set(prediction_stems)
    if not names or not predicted:
        return "no_image_index"
    if predicted == set(names):
        return "full_set"
    for stride in (2, 4):
        for offset in range(stride):
            subset = set(names[offset::stride])
            if predicted == subset:
                return f"stride_{stride}_offset_{offset}"
    overlaps = []
    for stride in (2, 4):
        subset = set(names[::stride])
        overlaps.append(f"overlap_stride_{stride}={len(predicted & subset)}/{len(predicted)}")
    return "not_a_fixed_stride;" + ";".join(overlaps)


def archived_macro(model: str, tier: str) -> str:
    """Read the Weighted Mean Macro-F1 already stored for this model and tier."""
    path = ROOT / "Results" / "F1" / model / f"f1_scores_{tier}.csv"
    if not path.exists():
        return ""
    for row in read_csv(path):
        if row.get("Diagram") == "Weighted Mean":
            return row.get("Macro_F1", "")
    return ""


def mean(values: List[float]) -> str:
    """Format a mean, or blank when there is nothing to average."""
    if not values:
        return ""
    return f"{sum(values) / len(values):.3f}"


def audit_models(cache: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Score every model and tier under the frozen policies."""
    coverage_rows: List[Dict[str, Any]] = []
    gap_rows: List[Dict[str, Any]] = []
    per_diagram: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}

    for spec in TIERS:
        diagrams = cache[spec["tier"]]
        images = image_stems(spec)
        for model in MODELS:
            pred_dir = ROOT / spec["pred"] / model
            pred_paths = {path.stem: path for path in pred_dir.glob("*.json")} if pred_dir.exists() else {}
            gt_stems = {item["stem"] for item in diagrams}
            status_counts: Dict[str, int] = defaultdict(int)
            flag_counts: Dict[str, int] = defaultdict(int)
            strict_scores: List[float] = []
            conditional_scores: List[float] = []
            lenient_scores: List[float] = []
            model_diagrams: Dict[str, Dict[str, Any]] = {}

            for item in diagrams:
                path = pred_paths.get(item["stem"])
                if path is None:
                    status, payload, flags = "missing", None, {"fenced": 0, "preamble": 0, "refusal": 0, "truncated": 0}
                else:
                    status, payload, flags = classify_file(path)
                for flag, value in flags.items():
                    flag_counts[flag] += value
                # Strict and conditional accept only JSON that parses directly.
                # Fence extraction is reported as lenient and does not raise the
                # strict score.
                direct = status in CONDITIONAL_STATUSES
                try:
                    scored = score_prediction(item["json"], payload if direct else None)
                except (TypeError, ValueError, AttributeError):
                    # A file can be JSON and still violate the ER schema, for
                    # example when an attribute list contains objects. The
                    # historical comparator cannot score that file, so strict
                    # treats it as an empty prediction.
                    status = "invalid_structure"
                    direct = False
                    scored = score_prediction(item["json"], None)
                status_counts[status] += 1
                strict_scores.append(scored["macro"])
                if direct:
                    conditional_scores.append(scored["macro"])
                    lenient_scores.append(scored["macro"])
                elif status in LENIENT_STATUSES:
                    try:
                        lenient_scores.append(score_prediction(item["json"], payload)["macro"])
                    except (TypeError, ValueError, AttributeError):
                        status_counts[status] -= 1
                        status = "invalid_structure"
                        status_counts[status] += 1
                model_diagrams[item["stem"]] = {
                    "status": status,
                    "scored": scored,
                    "included_conditional": direct,
                    "included_lenient": status in LENIENT_STATUSES,
                }
                if status not in CONDITIONAL_STATUSES:
                    gap_rows.append(
                        {
                            "model": model,
                            "tier": spec["tier"],
                            "diagram": item["stem"],
                            "status": status,
                        }
                    )

            unmatched = sorted(set(pred_paths) - gt_stems)
            coverage_rows.append(
                {
                    "model": model,
                    "tier": spec["tier"],
                    "n_gt": len(diagrams),
                    "n_prediction_files": len(pred_paths),
                    "n_matched_valid": status_counts["valid_json"] + status_counts["empty_schema"],
                    "n_missing": status_counts["missing"],
                    "n_invalid": status_counts["invalid_json"] + status_counts["invalid_structure"],
                    "n_truncated": status_counts["truncated"],
                    "n_refusal": status_counts["refusal"],
                    "n_lenient_only": status_counts["lenient_extracted"],
                    "n_empty_schema": status_counts["empty_schema"],
                    "n_fenced_flag": flag_counts["fenced"],
                    "n_preamble_flag": flag_counts["preamble"],
                    "n_unmatched_predictions": len(unmatched),
                    "strict_macro": mean(strict_scores),
                    "conditional_macro": mean(conditional_scores),
                    "lenient_macro": mean(lenient_scores),
                    "archived_weighted_macro": archived_macro(model, spec["tier"]),
                    "sampling": describe_sampling(images, pred_paths),
                }
            )
            per_diagram[(model, spec["tier"])] = model_diagrams
    return coverage_rows, gap_rows, per_diagram


def category_table(cache: Dict[str, List[Dict[str, Any]]], per_diagram: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Build Web/Instructional and medium+hard category F1 under both policies."""
    groups = {
        "Web/Instr": ["Web", "Instructional"],
        "Synthetic-med-hard": ["med", "hard"],
    }
    rows = []
    for group_name, tier_names in groups.items():
        for policy in (POLICY_STRICT, POLICY_CONDITIONAL):
            model_macros = []
            category_means = []
            support = {label: {"diagrams": 0, "elements": 0} for label, _key in CATEGORIES}
            for tier in tier_names:
                for item in cache[tier]:
                    for label, key in CATEGORIES:
                        gt_count = item["fresh"][key]
                        if gt_count > 0:
                            support[label]["diagrams"] += 1
                            support[label]["elements"] += gt_count
            for model in MODELS:
                numerators = {label: 0.0 for label, _key in CATEGORIES}
                denominators = {label: 0.0 for label, _key in CATEGORIES}
                macros = []
                for tier in tier_names:
                    for item in cache[tier]:
                        record = per_diagram[(model, tier)][item["stem"]]
                        if policy == POLICY_CONDITIONAL and not record["included_conditional"]:
                            continue
                        scored = record["scored"]
                        macros.append(scored["macro"])
                        for label, _key in CATEGORIES:
                            gt_count = scored[f"gt_{label}"]
                            if gt_count > 0:
                                numerators[label] += gt_count * scored[label]
                                denominators[label] += gt_count
                if not macros:
                    continue
                model_macros.append(sum(macros) / len(macros))
                category_means.append(
                    {label: (numerators[label] / denominators[label]) if denominators[label] else "" for label, _key in CATEGORIES}
                )
            averaged = {}
            for label, _key in CATEGORIES:
                present = [item[label] for item in category_means if item[label] != ""]
                averaged[label] = mean(present)
            row_values = [float(averaged[label]) for label, _key in CATEGORIES if averaged[label] != ""]
            rows.append(
                {
                    "group": group_name,
                    "policy": policy,
                    "n_models": len(model_macros),
                    "mean_diagram_macro": mean(model_macros),
                    "mean_of_category_rows": mean(row_values),
                    **{label: averaged[label] for label, _key in CATEGORIES},
                    **{f"support_diagrams_{label}": support[label]["diagrams"] for label, _key in CATEGORIES},
                    **{f"support_elements_{label}": support[label]["elements"] for label, _key in CATEGORIES},
                }
            )
    return rows


def shared_subset(cache: Dict[str, List[Dict[str, Any]]], per_diagram: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Compare Llama and Claude on identical diagrams under both policies."""
    left = "Llama-3.2"
    right = "Claude-4.6-sonnet"
    rows = []
    for spec in TIERS:
        tier = spec["tier"]
        both = []
        llama_strict = []
        claude_strict = []
        llama_conditional = []
        claude_conditional = []
        for item in cache[tier]:
            llama = per_diagram[(left, tier)][item["stem"]]
            claude = per_diagram[(right, tier)][item["stem"]]
            llama_strict.append(llama["scored"]["macro"])
            claude_strict.append(claude["scored"]["macro"])
            if llama["included_conditional"]:
                llama_conditional.append(llama["scored"]["macro"])
            if claude["included_conditional"]:
                claude_conditional.append(claude["scored"]["macro"])
            if llama["included_conditional"] and claude["included_conditional"]:
                both.append((llama["scored"]["macro"], claude["scored"]["macro"]))
        llama_both = [pair[0] for pair in both]
        claude_both = [pair[1] for pair in both]
        rows.append(
            {
                "tier": tier,
                "n_gt": len(cache[tier]),
                "n_both_valid": len(both),
                "llama_strict": mean(llama_strict),
                "claude_strict": mean(claude_strict),
                "llama_minus_claude_strict": (
                    f"{(sum(llama_strict) / len(llama_strict)) - (sum(claude_strict) / len(claude_strict)):.3f}"
                    if llama_strict and claude_strict
                    else ""
                ),
                "llama_conditional_own": mean(llama_conditional),
                "claude_conditional_own": mean(claude_conditional),
                "llama_shared_valid": mean(llama_both),
                "claude_shared_valid": mean(claude_both),
                "llama_minus_claude_shared_valid": (
                    f"{(sum(llama_both) / len(llama_both)) - (sum(claude_both) / len(claude_both)):.3f}"
                    if both
                    else ""
                ),
            }
        )
    return rows


def table1_check() -> List[Dict[str, Any]]:
    """Check the published instance total against unique-schema rows."""
    rows = read_csv(AGGREGATE_CSV)
    parsed = []
    for row in rows:
        if row["Diagram Number"] == "Total Sum":
            continue
        parsed.append(row)
    instances = sum(int(row["num_diagrams"]) for row in parsed)
    unique = sum(int(row["num_diagrams"]) for row in parsed if row["Diagram Number"] in UNIQUE_ROWS)
    total_row = next(row for row in rows if row["Diagram Number"] == "Total Sum")
    return [
        {
            "instances_from_rows": instances,
            "instances_total_cell": int(total_row["num_diagrams"]),
            "unique_schemas": unique,
            "unary_total": int(total_row["unary"]),
            "note": "Unique schemas are the base rows only. Biased, Random, Graphviz, and Silber rows are extra instances.",
        }
    ]


def gt_drift(cache: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Report diagrams whose archived stats CSV disagrees with current JSON."""
    rows = []
    for spec in TIERS:
        mismatched = 0
        json_entities = 0
        stats_entities = 0
        for item in cache[spec["tier"]]:
            json_entities += item["fresh"]["entities"]
            if item["stats_entities"] is None:
                continue
            stats_entities += item["stats_entities"]
            if item["stats_entities"] != item["fresh"]["entities"]:
                mismatched += 1
        rows.append(
            {
                "tier": spec["tier"],
                "json_entities": json_entities,
                "stats_entities": stats_entities,
                "diagrams_with_entity_drift": mismatched,
            }
        )
    return rows


def write_archived_diagram_means() -> None:
    """Save the mean of per-diagram Macro-F1, which is the table value.

    The Weighted Mean cell in some synthetic files does not equal that mean.
    The paper table follows the diagram rows.
    """
    rows = []
    tiers = [
        "easy", "med", "hard", "Web", "Web-Silber",
        "Instructional", "Instructional-Silber", "Schema", "extreme", "IsA",
    ]
    for model in MODELS:
        for tier in tiers:
            path = ROOT / "Results" / "F1" / model / f"f1_scores_{tier}.csv"
            if not path.exists():
                continue
            data = [
                float(row["Macro_F1"])
                for row in read_csv(path)
                if row.get("Diagram") not in {"Weighted Mean", "Mean"} and row.get("Macro_F1")
            ]
            summary = next(
                (row.get("Macro_F1", "") for row in read_csv(path) if row.get("Diagram") == "Weighted Mean"),
                "",
            )
            if not data:
                continue
            diagram_mean = sum(data) / len(data)
            rows.append(
                {
                    "model": model,
                    "tier": tier,
                    "n": len(data),
                    "diagram_mean_macro": f"{diagram_mean:.3f}",
                    "summary_cell_macro": summary,
                    "summary_matches_diagram_mean": abs(diagram_mean - float(summary)) <= 0.01 if summary else False,
                }
            )
    write_csv(
        OUT_DIR / "diagram_macro.csv",
        rows,
        ["model", "tier", "n", "diagram_mean_macro", "summary_cell_macro", "summary_matches_diagram_mean"],
    )


def main() -> None:
    """Write the audit tables."""
    print("Loading ground truth...")
    cache = {spec["tier"]: load_tier_ground_truth(spec) for spec in TIERS}
    print("Scoring archived predictions...")
    coverage, gaps, per_diagram = audit_models(cache)
    categories = category_table(cache, per_diagram)
    shared = shared_subset(cache, per_diagram)
    write_csv(OUT_DIR / "coverage_and_macro.csv", coverage, list(coverage[0].keys()))
    write_csv(OUT_DIR / "category_f1.csv", categories, list(categories[0].keys()))
    write_csv(OUT_DIR / "llama_claude_shared.csv", shared, list(shared[0].keys()))
    write_csv(OUT_DIR / "table1_check.csv", table1_check(), list(table1_check()[0].keys()))
    drift = gt_drift(cache)
    write_csv(OUT_DIR / "gt_drift.csv", drift, list(drift[0].keys()))
    write_csv(
        OUT_DIR / "missing_predictions.csv",
        gaps,
        ["model", "tier", "diagram", "status"],
    )
    write_archived_diagram_means()
    print(f"Wrote audit tables to {OUT_DIR}")
    print(f"Missing or unusable predictions: {len(gaps)}")


if __name__ == "__main__":
    main()
