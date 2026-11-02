"""
f1score.py - Compute F1 scores from error counts and ground truth
"""
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict


# =====================================================
# Configuration
# =====================================================

CATEGORIES = [
    "Entities", "RegAttr", "Multivalued", "Derived", "Composite",
    "PrimaryKeys", "Relationships", "Cardinality", "WeakEntities",
    "WeakRelationships", "IsA", "Nary"
]

GT_COLS = [f"GT_{c}" for c in CATEGORIES]
F1_COLS = [f"F1_{c}" for c in CATEGORIES]


# =====================================================
# F1 Computation Functions
# =====================================================

def compute_f1(gt: int, extra: int, missed: int, miscon: int) -> float:
    """
    Compute F1 score for a given category.
    Misconnections are treated as 0.5 FP and 0.5 FN.
    """
    f = 0.5
    if gt <= 0:
        return 0.0

    TP = max(0, gt - missed)
    FP = extra + (1 - f) * miscon
    FN = missed + f * miscon

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / gt if gt > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 3)


def compute_balanced_weighted_f1(row, gt_cols, f1_cols, balance_factor=0.3):
    """Compute balanced weighted F1 that limits influence of high-frequency categories."""
    gt_values = row[gt_cols].values
    f1_values = row[f1_cols].values
    
    non_zero_mask = gt_values > 0
    if not non_zero_mask.any():
        return 0.0
    
    gt_nonzero = gt_values[non_zero_mask]
    f1_nonzero = f1_values[non_zero_mask]
    
    freq_weights = gt_nonzero / gt_nonzero.sum()
    equal_weights = np.ones_like(freq_weights) / len(freq_weights)
    balanced_weights = balance_factor * freq_weights + (1 - balance_factor) * equal_weights
    
    return round(np.average(f1_nonzero, weights=balanced_weights), 3)


def compute_difficulty_weighted_f1(row, gt_cols, f1_cols, difficulty_weights):
    """Weight categories by their difficulty (lower avg performance = higher weight)."""
    gt_values = row[gt_cols].values
    f1_values = row[f1_cols].values
    
    non_zero_mask = gt_values > 0
    if not non_zero_mask.any():
        return 0.0
    
    active_difficulties = np.array([difficulty_weights.get(cat, 1.0) for cat in CATEGORIES])[non_zero_mask]
    f1_nonzero = f1_values[non_zero_mask]
    
    active_difficulties = active_difficulties / active_difficulties.sum()
    
    return round(np.average(f1_nonzero, weights=active_difficulties), 3)


def compute_macro_f1(row, gt_cols, f1_cols):
    """Compute macro F1: simple average of all categories with non-zero GT."""
    gt_values = row[gt_cols].values
    f1_values = row[f1_cols].values
    
    non_zero_mask = gt_values > 0
    if not non_zero_mask.any():
        return 0.0
    
    return round(f1_values[non_zero_mask].mean(), 3)


def estimate_category_difficulty(df_merged):
    """Estimate category difficulty based on average F1 performance."""
    df_no_sum = df_merged[df_merged["Diagram"] != "Sum"]
    difficulty_weights = {}
    
    for cat in CATEGORIES:
        mask = df_no_sum[f"GT_{cat}"] > 0
        if mask.any():
            avg_f1 = df_no_sum.loc[mask, f"F1_{cat}"].mean()
            difficulty_weights[cat] = 1.0 / (avg_f1 + 0.1)
        else:
            difficulty_weights[cat] = 1.0
    
    return difficulty_weights


# =====================================================
# Core F1 Computation
# =====================================================

def compute_f1_scores(df_gt: pd.DataFrame, df_err: pd.DataFrame, model: str, 
                      difficulty: str, output_dir: Path) -> pd.DataFrame:
    """
    Merge ground truth and error data, compute F1 per category and save results.
    
    Args:
        df_gt: Ground truth DataFrame
        df_err: Error count DataFrame
        model: Model name
        difficulty: Difficulty level
        output_dir: Directory to save F1 results
    
    Returns:
        DataFrame with F1 scores
    """
    # Merge data
    df_merged = pd.merge(df_err, df_gt, left_on="Diagram", right_on="Diagram Number")

    # Map GT columns
    df_merged["GT_Entities"] = df_merged["entities"]
    df_merged["GT_RegAttr"] = df_merged["attributes"]
    df_merged["GT_Multivalued"] = df_merged["multivalued"]
    df_merged["GT_Derived"] = df_merged["derived"]
    df_merged["GT_Composite"] = df_merged["composite"]
    df_merged["GT_PrimaryKeys"] = df_merged["primary_keys"]
    df_merged["GT_Relationships"] = df_merged["relationships"]
    df_merged["GT_Cardinality"] = df_merged["relationships"]
    df_merged["GT_WeakEntities"] = df_merged["weak_entities"]
    df_merged["GT_WeakRelationships"] = df_merged["weak_relationships"]
    df_merged["GT_IsA"] = df_merged["IsA"]
    df_merged["GT_Nary"] = df_merged["n_ary"]
    df_merged["Total"] = df_merged["total"]

    # Compute F1 per category
    df_merged["F1_Entities"] = df_merged.apply(lambda r: compute_f1(r["GT_Entities"], r["Ent-extra"], r["Ent-missed"], 0), axis=1)
    df_merged["F1_RegAttr"] = df_merged.apply(lambda r: compute_f1(r["GT_RegAttr"], r["Attr-extra"], r["Attr-missed"], r["Attr-miscon"]), axis=1)
    df_merged["F1_Multivalued"] = df_merged.apply(lambda r: compute_f1(r["GT_Multivalued"], 0, r["multival-missed"], 0), axis=1)
    df_merged["F1_Derived"] = df_merged.apply(lambda r: compute_f1(r["GT_Derived"], 0, r["derived-missed"], 0), axis=1)
    df_merged["F1_Composite"] = df_merged.apply(lambda r: compute_f1(r["GT_Composite"], 0, r["comp-missed"], 0), axis=1)
    df_merged["F1_PrimaryKeys"] = df_merged.apply(lambda r: compute_f1(r["GT_PrimaryKeys"], r["PrimKey-extra"], r["PrimKey-missed"], 0), axis=1)
    df_merged["F1_Relationships"] = df_merged.apply(lambda r: compute_f1(r["GT_Relationships"], r["Rel-extra"], r["Rel-missed"], r["Rel-miscon"]), axis=1)
    df_merged["F1_Cardinality"] = df_merged.apply(lambda r: compute_f1(r["GT_Cardinality"], 0, r["Cardinality"], 0), axis=1)
    df_merged["F1_WeakEntities"] = df_merged.apply(lambda r: compute_f1(r["GT_WeakEntities"], r["WeakEnt-extra"], r["WeakEnt-missed"], 0), axis=1)
    df_merged["F1_WeakRelationships"] = df_merged.apply(lambda r: compute_f1(r["GT_WeakRelationships"], r["WeakRel-extra"], r["WeakRel-missed"], 0), axis=1)
    df_merged["F1_IsA"] = df_merged.apply(lambda r: compute_f1(r["GT_IsA"], r["IsA-extra"], r["IsA-missed"], r["IsA-miscon"]), axis=1)
    df_merged["F1_Nary"] = df_merged.apply(lambda r: compute_f1(r["GT_Nary"], 0, r["N-ary-missed"], 0), axis=1)
    
    # Estimate difficulty weights
    difficulty_weights = estimate_category_difficulty(df_merged)
    
    # Compute various F1 aggregations
    df_merged["Macro_F1"] = df_merged.apply(lambda r: compute_macro_f1(r, GT_COLS, F1_COLS), axis=1)
    df_merged["Balanced_F1"] = df_merged.apply(lambda r: compute_balanced_weighted_f1(r, GT_COLS, F1_COLS, 0.3), axis=1)
    df_merged["Difficulty_F1"] = df_merged.apply(lambda r: compute_difficulty_weighted_f1(r, GT_COLS, F1_COLS, difficulty_weights), axis=1)
    df_merged["Simple_F1"] = df_merged[F1_COLS].mean(axis=1).round(3)
    
    # Inverse Weighted F1
    def inverse_weighted_f1(row):
        gt_values = row[GT_COLS].values
        f1_values = row[F1_COLS].values
        
        non_zero_gt_mask = gt_values > 0
        if not non_zero_gt_mask.any():
            return 0.0

        weights = np.zeros_like(gt_values, dtype=float)
        weights[non_zero_gt_mask] = 1.0 / gt_values[non_zero_gt_mask]
        
        total_weight = weights.sum()
        if total_weight == 0:
            return 0.0
        normalized_weights = weights / total_weight
        
        return round(np.average(f1_values, weights=normalized_weights), 3)

    df_merged["Inverse_Weighted_F1"] = df_merged.apply(inverse_weighted_f1, axis=1)
    df_merged["Freq_Weighted_F1"] = round(
        sum(df_merged[gt] * df_merged[f1] for gt, f1 in zip(GT_COLS, F1_COLS)) / df_merged[GT_COLS].sum(axis=1),
        3
    )

    # Build result table
    result_cols = ["Diagram"] + F1_COLS + [
        "Macro_F1", "Balanced_F1", "Difficulty_F1", 
        "Simple_F1", "Inverse_Weighted_F1", "Freq_Weighted_F1"
    ]
    result = df_merged[result_cols].copy()

    # Calculate means
    df_no_sum = df_merged[df_merged["Diagram"] != "Sum"]

    weighted_means = {
        f"F1_{cat}": round((df_no_sum[f"GT_{cat}"] * df_no_sum[f"F1_{cat}"]).sum() / df_no_sum[f"GT_{cat}"].sum(), 3)
        for cat in CATEGORIES if df_no_sum[f"GT_{cat}"].sum() > 0
    }
    
    weighted_means["Macro_F1"] = round(df_no_sum["Macro_F1"].mean(), 3)
    weighted_means["Balanced_F1"] = round(df_no_sum["Balanced_F1"].mean(), 3)
    weighted_means["Difficulty_F1"] = round(df_no_sum["Difficulty_F1"].mean(), 3)
    weighted_means["Simple_F1"] = round((df_no_sum["Total"] * df_no_sum["Simple_F1"]).sum() / df_no_sum["Total"].sum(), 3)
    weighted_means["Inverse_Weighted_F1"] = round(df_no_sum["Inverse_Weighted_F1"].mean(), 3)
    weighted_means["Freq_Weighted_F1"] = round((df_no_sum["Freq_Weighted_F1"] * df_no_sum["Total"]).sum() / df_no_sum["Total"].sum(), 3)
    weighted_means["Diagram"] = "Weighted Mean"
    result = pd.concat([result, pd.DataFrame([weighted_means])], ignore_index=True)

    means = {
        f"F1_{cat}": round(df_no_sum[f"F1_{cat}"].mean(), 3)
        for cat in CATEGORIES if df_no_sum[f"GT_{cat}"].sum() > 0
    }
    means["Macro_F1"] = round(df_no_sum["Macro_F1"].mean(), 3)
    means["Balanced_F1"] = round(df_no_sum["Balanced_F1"].mean(), 3)
    means["Difficulty_F1"] = round(df_no_sum["Difficulty_F1"].mean(), 3)
    means["Simple_F1"] = round(df_no_sum["Simple_F1"].mean(), 3)
    means["Inverse_Weighted_F1"] = round(df_no_sum["Inverse_Weighted_F1"].mean(), 3)
    means["Freq_Weighted_F1"] = round(df_no_sum["Freq_Weighted_F1"].mean(), 3)
    means["Diagram"] = "Mean"
    result = pd.concat([result, pd.DataFrame([means])], ignore_index=True)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / f"f1_scores_{difficulty}.csv", index=False)

    return result


# =====================================================
# Main Function
# =====================================================

def run_f1_scoring(models: List[str], difficulties: List[str], 
                   results_dir: str, gt_stats_dir: str) -> None:
    """
    Compute F1 scores from error counts and ground truth.
    
    Args:
        models: List of model names (e.g., ["openai/gpt-4o"])
        difficulties: List of difficulty levels (e.g., ["easy", "med"])
        results_dir: Directory containing error CSV files
        gt_stats_dir: Directory containing ground truth statistics
    """
    base_results = Path(results_dir)
    base_gt = Path(gt_stats_dir)
    
    summary_cols = ["Diagram"] + F1_COLS + [
        "Macro_F1", "Balanced_F1", "Difficulty_F1", 
        "Simple_F1", "Inverse_Weighted_F1", "Freq_Weighted_F1"
    ]
    total_f1 = pd.DataFrame(columns=summary_cols)

    for MODEL in models:
        model = MODEL
        print(f"Processing model: {model}")
        model_output_dir = base_results / model
        
        for difficulty in difficulties:
            print(f"  Difficulty: {difficulty}")
            
            # Read error data
            err_file = base_results / model / f"{difficulty}.csv"
            if not err_file.exists():
                print(f"    Warning: {err_file} not found, skipping")
                continue
            df_err = pd.read_csv(err_file)
            
            # Read ground truth
            if 'easy' in difficulty:
                gt_suffix = "easy"
            elif difficulty == "hard-text":
                gt_suffix = "hard"
            else:
                gt_suffix = difficulty

            gt_file = base_gt / f"sorted_stats_{gt_suffix}.csv"
            if not gt_file.exists():
                print(f"    Warning: {gt_file} not found, skipping")
                continue
            df_gt = pd.read_csv(gt_file)

            # Compute F1
            result = compute_f1_scores(df_gt, df_err, model, difficulty, model_output_dir)

            # Add to summary
            row = result[result["Diagram"] == "Weighted Mean"].copy()
            row["Diagram"] = f"{model} {difficulty}"
            total_f1 = pd.concat([total_f1, row], ignore_index=True)

    # Save global summary
    summary_path = base_results / "total_f1_scores.csv"
    total_f1.to_csv(summary_path, index=False)
    print(f"✅ F1 scoring complete! Summary saved to {summary_path}")


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
    
    DIFFICULTIES = ["easy-spanish", "easy-persian", "easy-romanian",'easy-german','easy-russian']
    
    run_f1_scoring(
        models=NON_REASONING.values(),
        difficulties=DIFFICULTIES,
        results_dir="Results/ERU-Agentic-2/",
        gt_stats_dir="Dataset/stats"
    )