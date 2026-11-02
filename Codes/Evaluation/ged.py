"""
ged.py - Compute graph-edit-distance-like structural scores for ER JSONs.
"""
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


def normalize_string(value: Any) -> str:
    """Normalize string-like values for robust structure matching."""
    if not isinstance(value, str):
        return str(value)
    return re.sub(r"[_\- .]", "", value.lower())


def normalize_list(items: Any) -> List[str]:
    """Return a normalized list for fields that may be scalar or list."""
    if isinstance(items, list):
        return [normalize_string(item) for item in items]
    if isinstance(items, str):
        return [normalize_string(items)]
    return [str(items)]


def normalize_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ER JSON entity and relationship names."""
    output = {"entities": {}, "relationships": []}

    for ent_name, ent_val in data.get("entities", {}).items():
        output["entities"][normalize_string(ent_name)] = ent_val

    for rel in data.get("relationships", []):
        normalized_rel = {}
        for key, value in rel.items():
            if key in {"entities", "subclasses"}:
                normalized_rel[key] = normalize_list(value)
            elif key in {"superclass", "rel_name", "name"}:
                normalized_rel[key] = normalize_string(value)
            else:
                normalized_rel[key] = value
        output["relationships"].append(normalized_rel)

    return output


def _add_value_nodes_and_edges(
    entity_name: str,
    field_prefix: str,
    values: Iterable[str],
    nodes: Set[str],
    edges: Set[Tuple[str, str, str]],
) -> None:
    """Add feature nodes and edges connected to an entity node."""
    for value in values or []:
        feature_node = f"{field_prefix}:{entity_name}:{normalize_string(value)}"
        nodes.add(feature_node)
        edges.add((f"E:{entity_name}", "has", feature_node))


def extract_graph_signature(data: Dict[str, Any]) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    """
    Convert normalized ER JSON to graph signature sets.

    The signature includes:
    - Nodes: entities, attributes, keys, relationship/ISA nodes.
    - Edges: entity-feature links, relationship membership, ISA links.
    """
    nodes: Set[str] = set()
    edges: Set[Tuple[str, str, str]] = set()

    for ent_name, ent_val in data.get("entities", {}).items():
        entity_node = f"E:{ent_name}"
        nodes.add(entity_node)

        _add_value_nodes_and_edges(ent_name, "A", normalize_list(ent_val.get("attributes", [])), nodes, edges)
        _add_value_nodes_and_edges(ent_name, "PK", normalize_list(ent_val.get("primary_keys", [])), nodes, edges)
        _add_value_nodes_and_edges(ent_name, "MV", normalize_list(ent_val.get("multivalued", [])), nodes, edges)
        _add_value_nodes_and_edges(ent_name, "DR", normalize_list(ent_val.get("derived", [])), nodes, edges)

        composite = ent_val.get("composite", {}) or {}
        if isinstance(composite, dict):
            for comp_name, sub_attrs in composite.items():
                comp_node = f"CP:{ent_name}:{normalize_string(comp_name)}"
                nodes.add(comp_node)
                edges.add((entity_node, "has_composite", comp_node))
                for sub_attr in normalize_list(sub_attrs):
                    sub_node = f"CSA:{ent_name}:{normalize_string(comp_name)}:{sub_attr}"
                    nodes.add(sub_node)
                    edges.add((comp_node, "has_subattr", sub_node))

    for rel in data.get("relationships", []):
        if "entities" in rel:
            rel_name = normalize_string(rel.get("name", ""))
            rel_node = f"R:{rel_name}"
            nodes.add(rel_node)
            for ent_name in normalize_list(rel.get("entities", [])):
                nodes.add(f"E:{ent_name}")
                edges.add((rel_node, "connects", f"E:{ent_name}"))

            if rel.get("cardinality"):
                card_node = f"CARD:{rel_name}:{normalize_string(rel.get('cardinality'))}"
                nodes.add(card_node)
                edges.add((rel_node, "has_cardinality", card_node))
        else:
            isa_name = normalize_string(rel.get("rel_name", ""))
            isa_node = f"ISA:{isa_name}"
            nodes.add(isa_node)
            superclass = normalize_string(rel.get("superclass", ""))
            if superclass:
                nodes.add(f"E:{superclass}")
                edges.add((isa_node, "superclass", f"E:{superclass}"))
            for subclass in normalize_list(rel.get("subclasses", [])):
                nodes.add(f"E:{subclass}")
                edges.add((isa_node, "subclass", f"E:{subclass}"))

    return nodes, edges


def compute_ged(gt_json: Dict[str, Any], test_json: Dict[str, Any]) -> Tuple[int, float]:
    """Compute raw and normalized graph edit distance from graph signatures."""
    gt_nodes, gt_edges = extract_graph_signature(gt_json)
    test_nodes, test_edges = extract_graph_signature(test_json)

    node_edits = len(gt_nodes.symmetric_difference(test_nodes))
    edge_edits = len(gt_edges.symmetric_difference(test_edges))
    raw_ged = node_edits + edge_edits

    gt_size = len(gt_nodes) + len(gt_edges)
    normalized_ged = (raw_ged / gt_size) if gt_size > 0 else 0.0
    return raw_ged, normalized_ged


def run_ged_evaluation(
    models: List[str],
    difficulties: List[str],
    dataset_dir: str,
    experiments_dir: str,
    results_dir: str,
    experiment_prefix: str = "Graphviz-",
) -> None:
    """Compute GED metrics for each model/difficulty and save per-run + summary CSVs."""
    dataset_root = Path(dataset_dir)
    experiments_root = Path(experiments_dir)
    results_root = Path(results_dir)
    summary_rows: List[Dict[str, Any]] = []

    for model in models:
        print(f"Processing model: {model}")
        model_output_dir = results_root / model
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for difficulty in difficulties:
            print(f"  Difficulty: {difficulty}")
            gt_dir = dataset_root / difficulty.replace("-text", "")
            pred_dir = experiments_root / f"{experiment_prefix}{difficulty}" / model

            if not gt_dir.exists() or not pred_dir.exists():
                print(f"    Warning: Missing directory for {model} / {difficulty}. Skipping.")
                continue

            row_details: List[Dict[str, Any]] = []
            for gt_file in sorted(gt_dir.glob("*.json")):
                pred_file = pred_dir / gt_file.name
                if not pred_file.exists():
                    continue

                try:
                    gt_json = normalize_json(json.loads(gt_file.read_text()))
                    pred_json = normalize_json(json.loads(pred_file.read_text()))
                except json.JSONDecodeError:
                    print(f"    Warning: JSON decode failed for {gt_file.name}. Skipping.")
                    continue

                raw_ged, norm_ged = compute_ged(gt_json, pred_json)
                row_details.append(
                    {
                        "Diagram": gt_file.stem,
                        "Raw_GED": raw_ged,
                        "Normalized_GED": round(norm_ged, 4),
                    }
                )

            if not row_details:
                continue

            details_path = model_output_dir / f"ged_{difficulty}.csv"
            with details_path.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["Diagram", "Raw_GED", "Normalized_GED"])
                writer.writeheader()
                writer.writerows(row_details)

            avg_raw = round(sum(r["Raw_GED"] for r in row_details) / len(row_details), 3)
            avg_norm = round(sum(r["Normalized_GED"] for r in row_details) / len(row_details), 4)
            summary_rows.append(
                {
                    "Model": model,
                    "Difficulty": difficulty,
                    "Average_Raw_GED": avg_raw,
                    "Average_Normalized_GED": avg_norm,
                    "Samples": len(row_details),
                }
            )

    summary_path = results_root / "total_ged_scores.csv"
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "Model",
                "Difficulty",
                "Average_Raw_GED",
                "Average_Normalized_GED",
                "Samples",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"✅ GED evaluation complete! Summary saved to {summary_path}")
