"""
compare.py - Compare ground truth and test JSONs to identify differences
"""
import json
import os
import re
from typing import Any, Dict, List
from pathlib import Path


# =====================================================
# Utility Functions
# =====================================================

def normalize_string(s: str) -> str:
    """Normalize a string by lowering case and removing underscores, dashes, spaces, and dots."""
    if not isinstance(s, str):
        return s
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
            normalized[key] = value
    return normalized


def normalize_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize JSON structure for consistent comparison."""
    output = {"entities": {}, "relationships": []}

    for ent_name, ent_val in data.get("entities", {}).items():
        norm_name = normalize_string(ent_name)
        output["entities"][norm_name] = normalize_entity(ent_val)

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

    return output


def list_diff(gt_list: List[str], test_list: List[str], label: str) -> List[str]:
    """Compare two lists and return human-readable differences."""
    diffs = []
    gt_set, test_set = set(gt_list or []), set(test_list or [])
    for item in gt_set - test_set:
        diffs.append(f"{label} '{item}' missing in test JSON.")
    for item in test_set - gt_set:
        diffs.append(f"{label} '{item}' extra in test JSON.")
    return diffs


def translation(diagram, mapping):
    """Translate entity and attribute names based on a provided mapping."""
    for new_i in range(len(mapping['translated_text'])):
        new = mapping['original_text'][new_i]
        old = normalize_string(mapping['translated_text'][new_i])
        diagram = re.sub(rf'\b{re.escape(old)}\b', new, diagram)
    return diagram


# =====================================================
# Comparison Functions
# =====================================================

def compare_entities(gt_entities: Dict[str, Any], test_entities: Dict[str, Any]) -> List[str]:
    """Compare entities between ground truth and test JSON."""
    diffs, diff_missed, diff_extra = [], [], []

    gt_keys, test_keys = set(gt_entities.keys()), set(test_entities.keys())

    # Entities missing
    for ent in gt_keys - test_keys:
        gt = gt_entities[ent]
        if gt.get("weak"):
            diffs.append(f"Entity '{ent}' weak flag missed.")
        diffs.append(f"Missed entity '{ent}' in test JSON.")

        diff_missed.extend([(a, ent) for a in gt.get("attributes", [])])
        diffs += list_diff(gt.get("primary_keys"), [], f"Entity '{ent}' primary key")
        diffs += list_diff(gt.get("multivalued"), [], f"Entity '{ent}' multivalued attribute")
        diffs += list_diff(gt.get("derived"), [], f"Entity '{ent}' derived attribute")

        for comp in gt.get("composite", {}) or {}:
            diffs.append(f"Entity '{ent}' composite attribute '{comp}' missing in test JSON.")

    # Extra entities
    for ent in test_keys - gt_keys:
        test = test_entities[ent]
        diffs.append(f"Extra entity '{ent}' in test JSON.")

        diff_extra.extend([(a, ent) for a in test.get("attributes", [])])
        diffs += list_diff([], test.get("primary_keys"), f"Entity '{ent}' primary key")
        diffs += list_diff([], test.get("multivalued"), f"Entity '{ent}' multivalued attribute")
        diffs += list_diff([], test.get("derived"), f"Entity '{ent}' derived attribute")

        if type(test.get("composite")) is dict:
            for comp in test.get("composite", {}):
                diffs.append(f"Extra composite attribute '{comp}' in entity '{ent}' of test JSON.")

    # Compare common entities
    for ent in sorted(gt_keys & test_keys):
        gt, test = gt_entities[ent], test_entities[ent]
        diff_missed.extend([(a, ent) for a in set(gt.get("attributes", [])) - set(test.get("attributes", []))])
        diff_extra.extend([(a, ent) for a in set(test.get("attributes", [])) - set(gt.get("attributes", []))])

        diffs += list_diff(gt.get("primary_keys"), test.get("primary_keys"), f"Entity '{ent}' primary key")
        diffs += list_diff(gt.get("multivalued"), test.get("multivalued"), f"Entity '{ent}' multivalued attribute")
        diffs += list_diff(gt.get("derived"), test.get("derived"), f"Entity '{ent}' derived attribute")

        gt_comp, test_comp = gt.get("composite", {}), test.get("composite", {})
        if test_comp is None:
            for comp in set(gt_comp):
                diffs.append(f"Entity '{ent}' composite attribute '{comp}' missing in test JSON.")
        else:
            for comp in set(gt_comp) - set(test_comp):
                diffs.append(f"Entity '{ent}' composite attribute '{comp}' missing in test JSON.")
            for comp in set(test_comp) - set(gt_comp):
                diffs.append(f"Extra composite attribute '{comp}' in entity '{ent}' of test JSON.")
            for comp in set(gt_comp) & set(test_comp):
                diffs += list_diff(gt_comp[comp], test_comp[comp], f"Entity '{ent}' composite '{comp}' sub-attribute")

        if bool(gt.get("weak")) != bool(test.get("weak")):
            diffs.append(
                f"Entity '{ent}' weak flag differs: ground truth={gt.get('weak')} vs test={test.get('weak')}"
            )

    # Handle misconnected attributes
    for item, ent in diff_missed:
        if (item, ent) not in diff_extra:
            diffs.append(f"Entity '{ent}' attribute '{item}' missing in test JSON.")
        else:
            diffs.append(f"Entity '{ent}' attribute '{item}' misconnected in test JSON.")

    for item, ent in diff_extra:
        if (item, ent) not in diff_missed:
            if all(item != d for d in diffs):
                diffs.append(f"Entity '{ent}' attribute '{item}' extra in test JSON.")

    return diffs


def compare_relationships(gt_rels: List[Dict[str, Any]], test_rels: List[Dict[str, Any]]) -> List[str]:
    """Compare relationships between ground truth and test JSONs."""
    diffs, unmatched = [], test_rels.copy()

    for gt in gt_rels:
        matched = next(
            (test for test in unmatched if
             (gt.get("name") and test.get("name") == gt.get("name")) or
             (gt.get("rel_name") and test.get("rel_name") == gt.get("rel_name"))),
            None,
        )

        if matched:
            unmatched.remove(matched)

            if "entities" in gt and "entities" in matched:
                if sorted(gt.get("entities", [])) != sorted(matched.get("entities", [])):
                    if len(gt.get("entities", [])) > 2:
                        diffs.append(f"Relationship '{gt['name']}' N-ary changed: {gt['entities']} vs {matched['entities']}.")
                    else:
                        diffs.append(f"Relationship '{gt['name']}' entities changed: {gt['entities']} vs {matched['entities']}.")

                if gt.get("cardinality") != matched.get("cardinality"):
                    diffs.append(
                        f"Relationship '{gt.get('name')}' cardinality changed: "
                        f"{gt.get('cardinality')} vs {matched.get('cardinality')}."
                    )
                if bool(gt.get("weak")) != bool(matched.get("weak")):
                    diffs.append(f"Relationship '{gt['name']}' weak flag differs: ground truth={gt.get('weak')} vs test={matched.get('weak')}.")
            else:
                if sorted(gt.get("subclasses", [])) != sorted(matched.get("subclasses", [])):
                    diffs.append(f"ISA '{gt['rel_name']}' subclasses changed: {gt['subclasses']} vs {matched.get('subclasses', [])}.")
                if gt.get("superclass") != matched.get("superclass"):
                    diffs.append(f"ISA '{gt['rel_name']}' superclass changed: {gt['superclass']} vs {matched.get('superclass')}.")

        else:
            if "entities" in gt:
                diffs.append(f"Relationship '{gt.get('name')}' missing in test JSON.")
                diffs.append(f"Relationship '{gt['name']}' entities changed: {gt['entities']} vs {[]}.")

                if len(gt.get("entities", [])) > 2:
                    diffs.append(f"Relationship '{gt['name']}' N-ary changed: {gt['entities']} vs {[]}.")
                
                if gt.get("cardinality"):
                   diffs.append(
                       f"Relationship '{gt.get('name')}' cardinality changed: "
                       f"{gt.get('cardinality')} vs {[]}."
                   )
                
                if gt.get("weak"):
                    diffs.append(f"Relationship '{gt['name']}' weak flag differs: ground truth={gt.get('weak')} vs test={not gt.get('weak')}.")

            else:
                diffs.append(f"ISA '{gt.get('rel_name')}' missing in test JSON.")
                
                if gt.get("superclass"):
                    diffs.append(f"ISA '{gt['rel_name']}' superclass changed: {gt['subclasses']} vs {[]}.")
                for sub in gt.get("subclasses", []):
                    diffs.append(f"ISA '{gt['rel_name']}' subclasses changed: {gt['superclass']} vs {[]}.")

    # Extra relationships
    for test in unmatched:
        if "entities" in test:
            diffs.append(f"Extra relationship {test} in test JSON.")
        else:
            diffs.append(f"Extra ISA {test} in test JSON.")
            if test.get("superclass"):
                diffs.append(f"ISA '{test.get('rel_name')}' superclass changed: {[]} vs {test.get('superclass')}.")
            for sub in test.get("subclasses", []):
                diffs.append(f"ISA '{test.get('rel_name')}' subclasses changed: {[]} vs {test.get('superclass')}.")

    return diffs


# =====================================================
# Main Evaluation Function
# =====================================================

def run_comparison(models: List[str], difficulties: List[str], dataset_dir: str, 
                   results_dir: str, experiments_dir: str, experiment_prefix: str = "Graphviz") -> None:
    """
    Run comparison evaluation across all models and difficulty levels.
    
    Args:
        models: List of model names (e.g., ["openai/gpt-4o", "anthropic/claude-3.5-haiku"])
        difficulties: List of difficulty levels (e.g., ["easy", "med", "hard"])
        dataset_dir: Path to ground truth dataset directory
        results_dir: Path to save comparison results
        experiments_dir: Path to experiment outputs
        experiment_prefix: Prefix for experiment folders (default: "Graphviz")
    """
    for MODEL in models:
        model = MODEL
        print(f"Processing model: {model}")
        
        for difficulty in difficulties:
            print(f"  Difficulty: {difficulty}")
            result_file = Path(results_dir) / model / f"{difficulty}.txt"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            
            dataset_path = Path(dataset_dir) / difficulty.replace('-text', '')
            experiment_path = Path(experiments_dir) / f"{experiment_prefix}{difficulty}" / model
            
            with open(result_file, "w") as f:
                f.write("Comparison Results:\n\n")

            for filename in os.listdir(experiment_path):
                if not filename.endswith(".json"):
                    continue

                gt_path = dataset_path / filename
                test_path = experiment_path / filename

                try:
                    with open(gt_path) as f:
                        gt_json = normalize_json(json.load(f))
                    with open(test_path) as f:
                        test_json = normalize_json(json.load(f))

                except (json.JSONDecodeError, OSError) as e:
                    print(f"    Error reading {filename}: {e}")
                    continue

                diffs = []
                diffs += compare_entities(gt_json.get("entities", {}), test_json.get("entities", {}))
                diffs += compare_relationships(gt_json.get("relationships", []), test_json.get("relationships", []))

                with open(result_file, "a") as f:
                    f.write(f"Diagram {filename.replace('.json', '')}\n\n")
                    if diffs:
                        f.writelines(f"{d}\n" for d in diffs)
                    else:
                        f.write("No differences found.\n")
                    f.write("\n")
    
    print("✅ Comparison complete!")


if __name__ == "__main__":  
    # Example standalone usage
    MODELS = {
        "Claude-4.6-sonnet": "anthropic/claude-sonnet-4.6",
        "GPT-5.2": "openai/gpt-5.2-chat",
        "Gemini-3-pro": "google/gemini-3-pro",
        "Grok-2-vision": "x-ai/grok-2-vision-latest",
        "Qwen-3.6": "qwen/qwen3.6-vl-32b-instruct",
        "Llama-3.2": "meta-llama/llama-3.2-90b-vision-instruct",
        "Gemma-3": "google/gemma-3-27b-it",
        "GLM-4.5": "z-ai/glm-4.5v",
    }

    DIFFICULTIES = ["easy", "med", "hard"]
    
    run_comparison(
        models=MODELS.keys(),
        difficulties=DIFFICULTIES,
        dataset_dir="Dataset/Generated/Graphviz-JSON/",
        results_dir="Results/F1/",
        experiments_dir="Experiments/Graphviz/",
        experiment_prefix="Graphviz-"
    )