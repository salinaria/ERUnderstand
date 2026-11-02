"""
Unified, simplified evaluation runner.
Only ONE function call needed.
"""

from pathlib import Path
from typing import List
import sys

# Import evaluation modules
from compare import run_comparison
from measure import run_measurement
from f1score import run_f1_scoring
from bleu import run_bleu_evaluation
from ged import run_ged_evaluation


# =====================================================
# Unified Config
# =====================================================

class EvaluationConfig:
    def __init__(
        self,
        models: List[str],
        difficulties: List[str],
        dataset_dir: str,
        experiments_dir: str,
        results_dir: str,
        gt_stats_dir: str = "Dataset/stats",
        experiment_prefix: str = "Graphviz"
    ):
        self.models = models
        self.difficulties = difficulties
        self.dataset_dir = dataset_dir
        self.experiments_dir = experiments_dir
        self.results_dir = results_dir
        self.gt_stats_dir = gt_stats_dir
        self.experiment_prefix = experiment_prefix

    def validate(self) -> bool:
        paths = [self.dataset_dir, self.experiments_dir, self.gt_stats_dir]
        ok = True
        for p in paths:
            if not Path(p).exists():
                print(f"❌ Missing directory: {p}")
                ok = False
        return ok


# =====================================================
# Main unified API
# =====================================================

def run_all_evaluations(
    models: List[str],
    difficulties: List[str],
    dataset_dir: str,
    experiments_dir: str,
    results_dir: str,
    gt_stats_dir: str = "Dataset/stats",
    experiment_prefix: str = "Graphviz-",
    steps: List[str] = None
):
    """
    A fully unified pipeline. Just call this ONCE.
    
    Example:
        run_all_evaluations(
            models=["openai/gpt-4o", "google/gemini"],
            difficulties=["easy", "hard"],
            dataset_dir="Dataset/Generated",
            experiments_dir="Experiments/",
            results_dir="Results/"
        )
    """
    config = EvaluationConfig(
        models=models,
        difficulties=difficulties,
        dataset_dir=dataset_dir,
        experiments_dir=experiments_dir,
        results_dir=results_dir,
        gt_stats_dir=gt_stats_dir,
        experiment_prefix=experiment_prefix
    )

    if steps is None:
        steps = ["compare", "measure", "f1", "bleu", "ged"]

    if not config.validate():
        print("❌ Invalid configuration. Fix paths.")
        return

    print("=" * 70)
    print("UNIFIED EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Models: {config.models}")
    print(f"Difficulties: {config.difficulties}")
    print(f"Steps: {steps}")
    print("=" * 70)

    # ---------------- STEP 1 ----------------
    if "compare" in steps:
        print("\n=== STEP 1: COMPARISON ===")
        try:
            run_comparison(
                models=config.models,
                difficulties=config.difficulties,
                dataset_dir=config.dataset_dir,
                results_dir=config.results_dir,
                experiments_dir=config.experiments_dir,
                experiment_prefix=config.experiment_prefix
            )
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            return

    # ---------------- STEP 2 ----------------
    if "measure" in steps:
        print("\n=== STEP 2: MEASUREMENT ===")
        try:
            run_measurement(
                models=config.models,
                difficulties=config.difficulties,
                results_dir=config.results_dir
            )
        except Exception as e:
            print(f"❌ Measurement failed: {e}")
            return

    # ---------------- STEP 3 ----------------
    if "f1" in steps:
        print("\n=== STEP 3: F1 SCORING ===")
        try:
            run_f1_scoring(
                models=config.models,
                difficulties=config.difficulties,
                results_dir=config.results_dir,
                gt_stats_dir=config.gt_stats_dir
            )
        except Exception as e:
            print(f"❌ F1 Scoring failed: {e}")
            return

    # ---------------- STEP 4 ----------------
    if "bleu" in steps:
        print("\n=== STEP 4: BLEU EVALUATION ===")
        try:
            bleu_dir = str(Path(config.results_dir) / "Bleu")
            run_bleu_evaluation(
                models=config.models,
                difficulties=config.difficulties,
                dataset_dir=config.dataset_dir,
                experiments_dir=config.experiments_dir,
                results_dir=bleu_dir,
                experiment_prefix=config.experiment_prefix
            )
        except Exception as e:
            print(f"❌ BLEU failed: {e}")
            return

    # ---------------- STEP 5 ----------------
    if "ged" in steps:
        print("\n=== STEP 5: GED EVALUATION ===")
        try:
            run_ged_evaluation(
                models=config.models,
                difficulties=config.difficulties,
                dataset_dir=config.dataset_dir,
                experiments_dir=config.experiments_dir,
                results_dir=config.results_dir,
                experiment_prefix=config.experiment_prefix,
            )
        except Exception as e:
            print(f"❌ GED failed: {e}")
            return

    print("\n" + "=" * 70)
    print("✅ ALL EVALUATIONS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {results_dir}")
    print("=" * 70)


# =====================================================
# Optional CLI (simplified)
# =====================================================

if __name__ == "__main__":
    print("❗ Use run_all_evaluations() directly in your python script.")
    # One results folder per model family.
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

    DIFFICULTIES = ["easy", "med", "hard", "med-random", "med-biased", "med-pk", "med-pk-empty", "med-persian", "translated-persian", "IsA", "extreme"]
    INPUT_ROOT="Dataset/Generated/Graphviz-JSON/"
    OUTPUT_ROOT="Experiments/Graphviz/"
    
    

    run_all_evaluations(
    models= MODELS.keys(),
    difficulties= DIFFICULTIES,
    dataset_dir=INPUT_ROOT,
    results_dir="Results/F1/",
    experiments_dir=OUTPUT_ROOT,    
    experiment_prefix="Graphviz-"
    )

    # Real-world evaluation commented out - directories don't exist
    # DIFFICULTIES = ["Instructional", "Instructional-Graphviz", "Instructional-Silber", "Web", "Web-Graphviz", "Web-Silber"]
    # INPUT_ROOT  = Path("Dataset/Real-world/JSON")  # adjust as needed
    # OUTPUT_ROOT = Path("Experiments") / "Real-world"

    # run_all_evaluations(
    # models= NON_REASONING.keys(),
    # difficulties= DIFFICULTIES,
    # dataset_dir=INPUT_ROOT,
    # results_dir="Results/F1/",
    # experiments_dir=OUTPUT_ROOT,    
    # experiment_prefix="Graphviz-"
    # )

    # run_all_evaluations(
    # models= REASONING.keys(),
    # difficulties= DIFFICULTIES,
    # dataset_dir=INPUT_ROOT,
    # results_dir="Results/F1/",
    # experiments_dir=OUTPUT_ROOT,    
    # experiment_prefix="Graphviz-"
    # )
