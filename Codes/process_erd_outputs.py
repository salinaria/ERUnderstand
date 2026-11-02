#!/usr/bin/env python3
"""
Convert relational schema JSON from erd_outputs into the project's Graphviz-JSON
format, render ERD PNGs, and compute per-diagram statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Codes/graphviz.py shadows the pip package when this script lives in Codes/.
import importlib.util
import sys

_codes_dir = str(Path(__file__).resolve().parent)
_removed_codes_path = False
if sys.path and sys.path[0] == _codes_dir:
    sys.path.pop(0)
    _removed_codes_path = True

_graphviz_spec = importlib.util.find_spec("graphviz")
if _graphviz_spec is None or _graphviz_spec.loader is None:
    raise ImportError("The graphviz Python package is required.")
_graphviz_module = importlib.util.module_from_spec(_graphviz_spec)
_graphviz_spec.loader.exec_module(_graphviz_module)
Digraph = _graphviz_module.Digraph

if _removed_codes_path:
    sys.path.insert(0, _codes_dir)

# Repository root (parent of Codes/)
REPO_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = REPO_ROOT / "ERUnderstand" / "erd_outputs"
JSON_OUT_DIR = REPO_ROOT / "ERUnderstand" / "Real-world" / "JSON" / "Schema-JSON"
PNG_OUT_DIR = REPO_ROOT / "ERUnderstand" / "Real-world" / "Diagrams" / "Schema-Graphviz"
STATS_DIR = REPO_ROOT / "ERUnderstand" / "stats"

# Strip dataset prefixes and the _erd suffix from raw filenames.
NAME_PREFIXES = ("spider_", "cypher_graph_", "bird_")


def normalize_diagram_name(filename: str) -> str:
    """Map raw schema filenames to clean diagram names used elsewhere in the dataset."""
    stem = Path(filename).stem
    if stem.endswith("_erd"):
        stem = stem[: -len("_erd")]
    for prefix in NAME_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem


def _is_relational_schema(schema: Dict[str, Any]) -> bool:
    """Return True when entities are stored as a list (raw erd_outputs format)."""
    entities = schema.get("entities")
    return isinstance(entities, list)


def _relationship_label(rel: Dict[str, Any], used_names: Set[str]) -> str:
    """Derive a unique relationship diamond label from a FK or graph edge."""
    rel_type = rel.get("type", "foreign_key")
    if rel_type == "foreign_key":
        base = rel.get("source_key", "references")
    else:
        base = re.sub(r"_RELATIONSHIP$", "", rel_type, flags=re.IGNORECASE)
        base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_").lower() or "relates"

    candidate = base
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def convert_relational_to_graphviz(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert array-based schema extraction JSON into the Graphviz-JSON dict format
    used by the rest of the ERUnderstanding pipeline.
    """
    if not _is_relational_schema(schema):
        return schema

    entities_out: Dict[str, Dict[str, Any]] = {}
    for entity in schema.get("entities", []):
        name = entity["name"]
        attrs = entity.get("attributes", [])
        attr_names = [a["name"] for a in attrs]
        primary_keys = [a["name"] for a in attrs if a.get("primary_key")]
        entities_out[name] = {
            "attributes": attr_names,
            "primary_keys": primary_keys,
            "weak": False,
        }

    relationships_out: List[Dict[str, Any]] = []
    used_names: Set[str] = set()
    for rel in schema.get("relationships", []):
        source = rel["source_entity"]
        target = rel["target_entity"]
        relationships_out.append(
            {
                "entities": [source, target],
                "cardinality": ["N", "1"],
                "name": _relationship_label(rel, used_names),
                "weak": False,
            }
        )

    return {"entities": entities_out, "relationships": relationships_out}


def create_png(
    context: str,
    entities: Dict[str, Any],
    relationships: List[Dict[str, Any]],
    out_dir: Path,
) -> Path:
    """Render an ER diagram PNG with Graphviz using project conventions."""
    dot = Digraph(format="png", engine="neato")
    dot.attr(overlap="false")

    for entity, data in entities.items():
        entity_style = "dashed" if data.get("weak", False) else "solid"
        dot.node(entity, entity, shape="rectangle", style=entity_style)

        primary_keys = set(data.get("primary_keys", []) or [])
        derived = set(data.get("derived", []) or [])
        multivalued = set(data.get("multivalued", []) or [])

        for attr in data.get("attributes", []) or []:
            if attr in primary_keys or attr in derived or attr in multivalued:
                continue
            attr_node = f"{entity}_{attr}"
            dot.node(attr_node, attr, shape="ellipse")
            dot.edge(attr_node, entity, dir="none")

        for pk in primary_keys:
            pk_node = f"{entity}_{pk}"
            dot.node(pk_node, f"<<u>{pk}</u>>", shape="ellipse")
            dot.edge(pk_node, entity, dir="none")

        for d_attr in derived:
            d_node = f"{entity}_{d_attr}"
            dot.node(d_node, d_attr, shape="ellipse", style="dashed")
            dot.edge(d_node, entity, dir="none")

        for m_attr in multivalued:
            m_node = f"{entity}_{m_attr}"
            dot.node(m_node, m_attr, shape="ellipse", peripheries="2")
            dot.edge(m_node, entity, dir="none")

        for comp_attr, sub_attrs in (data.get("composite") or {}).items():
            comp_node = f"{entity}_{comp_attr}"
            dot.node(comp_node, comp_attr, shape="ellipse")
            dot.edge(comp_node, entity, dir="none")
            for sub in sub_attrs:
                sub_node = f"{entity}_{comp_attr}_{sub}"
                dot.node(sub_node, sub, shape="ellipse")
                dot.edge(sub_node, comp_node, dir="none")

    rel_count = 0
    for rel in relationships:
        if "entities" in rel:
            rel_name = rel["name"]
            entities_involved = rel["entities"]
            cardinalities = rel["cardinality"]
            rel_style = "dashed" if rel.get("weak", False) else "solid"
            dot.node(rel_name, rel_name, shape="diamond", style=rel_style)
            for i, ent in enumerate(entities_involved):
                dot.edge(ent, rel_name, label=cardinalities[i], dir="none")
        else:
            subclasses = rel["subclasses"]
            rel_name = rel["rel_name"]
            superclass = rel["superclass"]
            rel_node = f"{rel_name}_{rel_count}"
            rel_count += 1
            dot.node(
                rel_node,
                rel_name,
                shape="polygon",
                sides="3",
                height="0.5",
                width="0.5",
                style="filled",
                fillcolor="lightgray",
            )
            for subclass in subclasses:
                dot.edge(subclass, rel_node, dir="none")
            dot.edge(rel_node, superclass, dir="none")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_base = out_dir / context
    dot.render(str(output_base), view=False, cleanup=True)
    return Path(f"{output_base}.png")


def analyze_diagram(name: str, diagram: Dict[str, Any]) -> Dict[str, Any]:
    """Compute structural statistics for one diagram (matches stats.py logic)."""
    stats = {
        "Diagram Number": name,
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
        "total": 0,
    }

    entities = diagram.get("entities", {})
    stats["entities"] = len(entities)

    for _entity_name, entity_info in entities.items():
        stats["attributes"] += len(entity_info.get("attributes", []))
        if "composite" in entity_info:
            stats["composite"] += len(entity_info["composite"])
            stats["attributes"] += len(entity_info["composite"])
            for attr in entity_info["composite"].keys():
                stats["attributes"] += len(entity_info["composite"][attr])
        if "multivalued" in entity_info:
            stats["multivalued"] += len(entity_info["multivalued"])
            stats["attributes"] += len(entity_info["multivalued"])
        if "derived" in entity_info:
            stats["derived"] += len(entity_info["derived"])
            stats["attributes"] += len(entity_info["derived"])
        if "primary_keys" in entity_info:
            stats["primary_keys"] += len(entity_info["primary_keys"])
        if entity_info.get("weak", False):
            stats["weak_entities"] += 1

    for rel in diagram.get("relationships", []):
        involved_entities = rel.get("entities", [])
        stats["attributes"] += len(rel.get("attributes", []))
        if len(involved_entities) == 2:
            if involved_entities[0] == involved_entities[1]:
                stats["unary"] += 1
        elif len(involved_entities) > 2 and "isa" not in rel.get("name", "").lower():
            stats["n_ary"] += 1

        if "rel_name" in rel:
            stats["IsA"] += 1
        else:
            stats["relationships"] += 1

        if rel.get("weak"):
            stats["weak_relationships"] += 1

    for key, value in stats.items():
        if key not in ("Diagram Number", "total"):
            stats["total"] += value

    return stats


def write_stats_csv(rows: List[Dict[str, Any]], stats_path: Path, sorted_path: Path) -> None:
    """Write stats CSV and a version sorted by total complexity."""
    fieldnames = list(rows[0].keys()) if rows else []
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sorted_rows = sorted(rows, key=lambda row: int(row["total"]))
    with open(sorted_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)


def update_aggregated_sums(stats_dir: Path) -> None:
    """Rebuild aggregated_sums.csv including the new Schema dataset."""
    import pandas as pd

    csv_files = sorted(stats_dir.glob("sorted_stats_*.csv"))
    results = []

    for file in csv_files:
        df = pd.read_csv(file)
        if "med" in file.name:
            for suffix in ("ium", "ium-Random", "ium-Biased"):
                sum_row = df[df["Diagram Number"] == "Sum"].copy()
                sum_row.insert(1, "num_diagrams", len(df) - 1)
                sum_row["Diagram Number"] = (
                    file.name.replace(".csv", "").replace("sorted_stats_", "") + suffix
                )
                sum_row = sum_row.drop(columns=["total"])
                results.append(sum_row)
        elif "Web" in file.name or "Instructional" in file.name:
            for suffix in ("", "-Silber", "-Graphviz"):
                sum_row = df[df["Diagram Number"] == "Sum"].copy()
                sum_row.insert(1, "num_diagrams", len(df) - 1)
                sum_row["Diagram Number"] = (
                    file.name.replace(".csv", "").replace("sorted_stats_", "") + suffix
                )
                sum_row = sum_row.drop(columns=["total"])
                results.append(sum_row)
        else:
            sum_row = df[df["Diagram Number"] == "Sum"].copy()
            sum_row.insert(1, "num_diagrams", len(df) - 1)
            sum_row["Diagram Number"] = file.name.replace(".csv", "").replace(
                "sorted_stats_", ""
            )
            sum_row = sum_row.drop(columns=["total"])
            results.append(sum_row)

    final_df = pd.concat(results, ignore_index=True)
    sort_order = ["Web", "Instructional", "Schema", "easy", "med", "hard", "extreme", "IsA"]
    final_df["sort_key"] = final_df["Diagram Number"].apply(
        lambda label: next((i for i, value in enumerate(sort_order) if value in label), len(sort_order))
    )
    final_df = final_df.sort_values(by=["sort_key", "Diagram Number"]).drop(columns=["sort_key"])

    total_sum = final_df.select_dtypes(include="number").sum()
    total_sum["Diagram Number"] = "Total Sum"
    final_df = pd.concat([final_df, pd.DataFrame([total_sum])], ignore_index=True)
    final_df.to_csv(stats_dir / "aggregated_sums.csv", index=False)


def process_all(input_dir: Path, json_out_dir: Path, png_out_dir: Path) -> Tuple[int, int, List[str]]:
    """Convert schemas, render diagrams, and return success/failure counts."""
    json_out_dir.mkdir(parents=True, exist_ok=True)
    png_out_dir.mkdir(parents=True, exist_ok=True)

    stats_rows: List[Dict[str, Any]] = []
    final_stats = {
        "Diagram Number": "Sum",
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
        "total": 0,
    }

    success = 0
    failures: List[str] = []
    source_files = sorted(input_dir.glob("*.json"))

    for source_path in source_files:
        diagram_name = normalize_diagram_name(source_path.name)
        try:
            with open(source_path, encoding="utf-8") as handle:
                raw_schema = json.load(handle)

            converted = convert_relational_to_graphviz(raw_schema)
            json_path = json_out_dir / f"{diagram_name}.json"
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(converted, handle, indent=2, ensure_ascii=False)

            create_png(
                diagram_name,
                converted["entities"],
                converted["relationships"],
                png_out_dir,
            )

            row = analyze_diagram(diagram_name, converted)
            stats_rows.append(row)
            for key in final_stats:
                if key != "Diagram Number":
                    final_stats[key] += row[key]
            success += 1
            print(f"[OK] {source_path.name} -> {diagram_name}")
        except Exception as exc:
            failures.append(f"{source_path.name}: {exc}")
            print(f"[FAIL] {source_path.name}: {exc}", file=sys.stderr)

    stats_rows.append(final_stats)
    stats_path = STATS_DIR / "stats_Schema.csv"
    sorted_path = STATS_DIR / "sorted_stats_Schema.csv"
    write_stats_csv(stats_rows, stats_path, sorted_path)
    update_aggregated_sums(STATS_DIR)

    return success, len(failures), failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Process erd_outputs schema JSON files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing raw *_erd.json schema files.",
    )
    parser.add_argument(
        "--json-out-dir",
        type=Path,
        default=JSON_OUT_DIR,
        help="Directory for converted Graphviz-JSON files.",
    )
    parser.add_argument(
        "--png-out-dir",
        type=Path,
        default=PNG_OUT_DIR,
        help="Directory for rendered Graphviz PNG diagrams.",
    )
    args = parser.parse_args()

    success, failure_count, failures = process_all(
        args.input_dir, args.json_out_dir, args.png_out_dir
    )
    print(f"\nDone: {success} succeeded, {failure_count} failed.")
    if failures:
        print("Failures:")
        for item in failures:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
