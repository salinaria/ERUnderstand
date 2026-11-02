"""
measure.py - Parse error text files and generate CSV summaries
"""
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


# =====================================================
# Configuration
# =====================================================

FIELDNAMES = [
    "Diagram",
    "Ent-extra", "Ent-missed",
    "Attr-missed", "Attr-extra", "Attr-miscon",
    "multival-missed", "derived-missed",
    "comp-extra", "comp-missed",
    "PrimKey-extra", "PrimKey-missed",
    "Rel-extra", "Rel-missed", "Rel-miscon", "Cardinality",
    "WeakRel-extra", "WeakRel-missed",
    "WeakEnt-extra", "WeakEnt-missed",
    "IsA-missed", "IsA-extra", "IsA-miscon", "N-ary-missed",
    "Sum"
]

PATTERNS = [
    # Composite attributes
    (re.compile(r"composite attribute '.+' missing"), "comp-missed"),
    (re.compile(r"Extra composite attribute '.+' in entity"), "comp-extra"),
    (re.compile(r"composite '.+' sub-attribute '.+' missing"), "comp-missed"),
    (re.compile(r"composite '.+' sub-attribute '.+' extra"), "comp-extra"),

    # Primary keys
    (re.compile(r"primary key '.+' extra in"), "PrimKey-extra"),
    (re.compile(r"primary key '.+' missing in"), "PrimKey-missed"),

    # Multivalued & Derived
    (re.compile(r"multivalued attribute '.+' missing"), "multival-missed"),
    (re.compile(r"derived attribute '.+' missing"), "derived-missed"),

    # Entities
    (re.compile(r"Extra entity '"), "Ent-extra"),
    (re.compile(r"Missed entity '"), "Ent-missed"),

    # Attributes
    (re.compile(r"attribute '.+' missing in"), "Attr-missed"),
    (re.compile(r"attribute '.+' extra in"), "Attr-extra"),
    (re.compile(r"attribute '.+' misconnected"), "Attr-miscon"),

    # Cardinality
    (re.compile(r"cardinality changed"), "Cardinality"),

    # Relationships
    (re.compile(r"Relationship '.+' N-ary changed"), "N-ary-missed"),
    (re.compile(r"Relationship '.+' entities changed"), "Rel-miscon"),
    (re.compile(r"Extra relationship "), "Rel-extra"),
    (re.compile(r"Relationship '.+' missing in"), "Rel-missed"),

    # ISA
    (re.compile(r"ISA .+ missing in"), "IsA-missed"),
    (re.compile(r"Extra ISA"), "IsA-extra"),
    (re.compile(r"ISA .+ changed"), "IsA-miscon"),

    # Weak flags
    (re.compile(r"Relationship '.+' weak flag differs: ground truth=False vs test=True"), "WeakRel-extra"),
    (re.compile(r"Relationship '.+' weak flag differs: ground truth=True vs test=False"), "WeakRel-missed"),
    (re.compile(r"Entity '.+' weak flag missed"), "WeakEnt-missed"),
    (re.compile(r"Entity '.+' weak flag differs: ground truth=False vs test=True"), "WeakEnt-extra"),
    (re.compile(r"Entity '.+' weak flag differs: ground truth=True vs test=False"), "WeakEnt-missed"),
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =====================================================
# Core Functions
# =====================================================

def parse_errors(txt_path: Path) -> List[Dict[str, Any]]:
    """Parse error text file and return structured error counts."""
    results = []
    counts = None

    try:
        with txt_path.open("r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                if line.startswith("Diagram "):
                    if counts:
                        counts["Sum"] = sum(counts[f] for f in FIELDNAMES if f not in ("Diagram", "Sum"))
                        results.append(counts)
                    counts = {f: 0 for f in FIELDNAMES}
                    counts["Diagram"] = line.split(" ", 1)[1].strip()
                    continue

                for pattern, field in PATTERNS:
                    if pattern.search(line):
                        counts[field] += 1
                        break
                else:
                    logging.warning(f"Unrecognized line in {txt_path}: {line}")

        if counts:
            counts["Sum"] = sum(counts[f] for f in FIELDNAMES if f not in ("Diagram", "Sum"))
            results.append(counts)

    except FileNotFoundError:
        logging.error(f"File not found: {txt_path}")

    return results


def write_csv(model: str, rows: List[Dict[str, Any]], csv_path: Path, 
              total_df: pd.DataFrame) -> pd.DataFrame:
    """Write parsed errors to CSV and update total dataframe."""
    if not rows:
        logging.warning(f"No rows parsed for {csv_path}. Skipping.")
        return total_df

    df = pd.DataFrame(rows, columns=FIELDNAMES)

    sum_row = df.sum(numeric_only=True)
    sum_row["Diagram"] = model
    df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    logging.info(f"Saved {csv_path}")

    sum_row.name = f"{csv_path.parent.name} {csv_path.stem}"
    return pd.concat([total_df, pd.DataFrame([sum_row])], ignore_index=True)


# =====================================================
# Main Function
# =====================================================

def run_measurement(models: List[str], difficulties: List[str], results_dir: str) -> None:
    """
    Parse comparison results and generate CSV summaries.
    
    Args:
        models: List of model names (e.g., ["openai/gpt-4o", "anthropic/claude-3.5-haiku"])
        difficulties: List of difficulty levels (e.g., ["easy", "med", "hard"])
        results_dir: Path to results directory containing .txt files
    """
    results_path = Path(results_dir)
    total_df = pd.DataFrame(columns=FIELDNAMES)

    for MODEL in models:
        model = MODEL
        print(f"Processing model: {model}")
        
        for difficulty in difficulties:
            txt_file = results_path / model / f"{difficulty}.txt"
            csv_file = results_path / model / f"{difficulty}.csv"

            logging.info(f"Processing {model} - {difficulty}")
            rows = parse_errors(txt_file)
            total_df = write_csv(model + difficulty, rows, csv_file, total_df)

    total_df.to_csv(results_path / "Total.csv", index=False)
    logging.info(f"Saved {results_path / 'Total.csv'}")
    print("✅ Measurement complete!")


if __name__ == "__main__":
    # Example standalone usage
    NON_REASONING = {
    "claude_3.5_haiku":                   "anthropic/claude-3.5-haiku",
    "gpt_4o":                             "openai/gpt-4o",
    "gemini_2.0_flash_001":               "google/gemini-2.0-flash-001",
    "qwen2.5_vl_72b_instruct":            "qwen/qwen2.5-vl-72b-instruct",
    "pixtral_large_2411":                 "mistralai/pixtral-large-2411",
    "llama_3.2_90b_vision_instruct":      "meta-llama/llama-3.2-90b-vision-instruct",
    }

    DIFFICULTIES = ["easy-spanish", "easy-persian", "easy-romanian"]
    
    run_measurement(
        models=NON_REASONING.values(),
        difficulties=DIFFICULTIES,
        results_dir="Results/ERU-Agentic-2/"
    )