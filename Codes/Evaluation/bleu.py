"""
bleu.py - Compute BLEU scores between ground truth and test JSONs
"""
import csv
import json
import re
from typing import Dict, Any, List
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu


# =====================================================
# Normalization Functions
# =====================================================

def normalize_string(s: str) -> str:
    """Normalize a string by lowering case and removing underscores, dashes, spaces, and dots."""
    if not isinstance(s, str):
        return str(s)
    return re.sub(r'[_\- .]', '', s.lower())


def normalize_list(items):
    if isinstance(items, list):
        return [normalize_string(item) for item in items]
    elif isinstance(items, str):
        return [normalize_string(items)]
    else:
        return [str(items)]


def normalize_entity(entity_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an entity dictionary recursively."""
    normalized = {}
    for key, value in entity_dict.items():
        if isinstance(value, list):
            normalized[key] = normalize_list(value)
        elif isinstance(value, dict):
            normalized[key] = {normalize_string(k): normalize_list(v) for k, v in value.items()}
        else:
            normalized[key] = normalize_string(value)
    return normalized


def normalize_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize JSON structure for consistent comparison."""
    output = {"entities": {}, "relationships": []}

    # Normalize entities
    for ent_name, ent_val in data.get("entities", {}).items():
        norm_name = normalize_string(ent_name)
        output["entities"][norm_name] = normalize_entity(ent_val)
    
    # Sort keys in entities
    for ent in output["entities"]:
        output["entities"][ent] = dict(sorted(output["entities"][ent].items()))
    
    # Sort attributes
    for ent in output["entities"]:
        if "attributes" in output["entities"][ent]:
            output["entities"][ent]["attributes"] = sorted(output["entities"][ent]["attributes"])

    # Normalize relationships
    for rel in data.get("relationships", []):
        new_rel = {}
        for key, val in rel.items():
            if key in {"entities", "subclasses"}:
                new_rel[key] = normalize_list(val)
            elif key in {"superclass", "rel_name", "name"}:
                new_rel[key] = normalize_string(val)
            else:
                new_rel[key] = val
        output["relationships"].append(new_rel)
    
    # Sort relationships
    output["relationships"] = sorted(output["relationships"], key=lambda x: (x.get("name", ""), x.get("rel_name", "")))

    # Sort entities in relationships
    for rel in output["relationships"]:
        if "entities" in rel:
            rel["entities"] = sorted(rel["entities"])
        if "subclasses" in rel:
            rel["subclasses"] = sorted(rel["subclasses"])

    return output


# =====================================================
# Token Conversion
# =====================================================

def json_to_tokens(data: Dict[str, Any]) -> list:
    """Flatten normalized JSON into a list of tokens for BLEU calculation."""
    tokens = []
    # Entities
    for ent_name, ent_val in data.get("entities", {}).items():
        tokens.append(ent_name)
        for k, v in ent_val.items():
            tokens.append(k)
            if isinstance(v, list):
                tokens.extend(v)
            else:
                tokens.append(str(v))
    # Relationships
    for rel in data.get("relationships", []):
        for k, v in rel.items():
            tokens.append(k)
            if isinstance(v, list):
                tokens.extend(v)
            else:
                tokens.append(str(v))
    return tokens


# =====================================================
# Main Function
# =====================================================

def run_bleu_evaluation(models: List[str], difficulties: List[str], 
                        dataset_dir: str, experiments_dir: str, 
                        results_dir: str, experiment_prefix: str = "Graphviz-") -> None:
    """
    Compute BLEU scores between ground truth and test JSONs.

    Path layout matches ged.py: experiments at {experiments_dir}/{prefix}{difficulty}/{model}/.
    Writes per-diagram CSV to {results_dir}/{model}/bleu_{difficulty}.csv.
    """
    results_root = Path(results_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(dataset_dir)
    experiments_root = Path(experiments_dir)

    summary_path = results_root / "total_bleu_scores.csv"
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

            result_file = model_output_dir / f"{difficulty}_bleu.txt"
            row_details: List[Dict[str, Any]] = []

            with result_file.open("w", encoding="utf-8") as handle:
                handle.write("BLEU Score Results:\n\n")

                for gt_path in sorted(gt_dir.glob("*.json")):
                    pred_path = pred_dir / gt_path.name
                    if not pred_path.exists():
                        continue

                    try:
                        gt_json = normalize_json(json.loads(gt_path.read_text(encoding="utf-8")))
                        pred_json = normalize_json(json.loads(pred_path.read_text(encoding="utf-8")))
                    except (json.JSONDecodeError, OSError) as exc:
                        print(f"    Error reading {gt_path.name}: {exc}")
                        continue

                    gt_tokens = json_to_tokens(gt_json)
                    test_tokens = json_to_tokens(pred_json)
                    bleu_score = corpus_bleu([[gt_tokens]], [test_tokens])
                    rounded = round(bleu_score, 4)
                    row_details.append({"Diagram": gt_path.stem, "BLEU": rounded})
                    handle.write(f"{gt_path.name}: BLEU Score = {rounded:.4f}\n")

            if not row_details:
                continue

            details_path = model_output_dir / f"bleu_{difficulty}.csv"
            with details_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=["Diagram", "BLEU"])
                writer.writeheader()
                writer.writerows(row_details)

            avg_bleu = round(sum(r["BLEU"] for r in row_details) / len(row_details), 4)
            with result_file.open("a", encoding="utf-8") as handle:
                handle.write(f"\nAverage BLEU: {avg_bleu:.4f}\n")

            summary_rows.append(
                {
                    "Model": model,
                    "Difficulty": difficulty,
                    "Average_BLEU": avg_bleu,
                    "Samples": len(row_details),
                }
            )

    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["Model", "Difficulty", "Average_BLEU", "Samples"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Legacy aggregate file for Graphviz-style workflows.
    legacy_path = results_root / "Bleu" / "average_bleu.csv"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    with legacy_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Model", "Difficulty", "Average BLEU"])
        for row in summary_rows:
            writer.writerow([row["Model"], row["Difficulty"], row["Average_BLEU"]])

    print(f"✅ BLEU evaluation complete! Summary saved to {summary_path}")


if __name__ == "__main__":
    # Example standalone usage
    MODELS = [
        "google/gemini-2.0-flash-001",
        "openai/gpt-4o",
        "anthropic/claude-3.5-haiku",
    ]
    
    DIFFICULTIES = ["easy-spanish", "easy-persian", "easy-romanian"]
    
    run_bleu_evaluation(
        models=MODELS,
        difficulties=DIFFICULTIES,
        dataset_dir="Dataset/Generated/Graphviz-JSON/",
        experiments_dir="Experiments/ERU-Agentic/",
        results_dir="Results/Bleu/",
        experiment_prefix="Graphviz"
    )