# ERUnderstand

Benchmark and evaluation codebase for **ERUnderstand**: Web, Instructional, Schema, and synthetic ER diagram understanding, with a unified structural evaluation pipeline (F1, BLEU, GED).

Ground truth and annotations use a **single JSON schema** across all tiers. That JSON format is the annotation guideline and the contract for model prompts (`Codes/Exp_API.py`).

**Paper naming (canonical):** Web · Instructional (Instr.) · Schema · Easy / Medium / Hard · Web-S / Instr.-S (Silberschatz variants). Folder names use the same terms.

---

## Repository layout

```
ERUnderstand/
├── ERUnderstand/          # Datasets, stats, annotations
│   ├── Real-world/        # Web, Instructional, Schema (JSON + diagram images)
│   ├── Generated/         # Synthetic tiers (Graphviz JSON + PNGs)
│   ├── erd_outputs/       # Raw Schema extractions (Spider / SQL / Cypher / BIRD)
│   ├── annotations/       # Inter-annotator pilot exports (Web + Instructional)
│   └── stats/             # Per-tier CSVs, aggregates, documentation
├── Codes/                 # Generation, conversion, and evaluation scripts
├── Experiments/           # Model prediction JSON (Real-world + Graphviz)
├── Results/F1/            # Comparison logs, per-model CSVs, aggregate score tables
└── Mermaid/               # Auxiliary diagram sources (not in main eval tiers)
```

---

## Benchmark tiers (paper names)

| Paper tier | Repo folder / path | Diagrams | Notation |
|------------|-------------------|--------:|----------|
| **Web** | `Web-JSON/`, `Web`, `Web-Graphviz`, `Web-Silber` | 100 schemas → 500 renders | Chen / Silberschatz (Web-S) / original |
| **Instructional** | `Instructional-JSON/`, `Instructional`, `Instructional-Graphviz`, `Instructional-Silber` | 48 schemas → 240 renders | Chen / Silberschatz (Instr.-S) / original |
| **Schema** | `Schema-JSON/`, `Schema-Graphviz` | 179 | Graphviz only |
| **Easy / Medium / Hard** | `Generated/Graphviz-JSON/{easy,med,hard}/` | 199 each | Graphviz |
| **High IS-A** | `Generated/Graphviz-JSON/IsA/` | 32 | Graphviz |

**Schema sources (179 diagrams):** Spider (164), classic SQL samples (9), Cypher graph templates (5), BIRD (1). Schema GT is **flat relational structure only**—no weak entities, ISA, composite/multivalued/derived attributes, or n-ary relationships in ground truth (SQL metadata limit). See `ERUnderstand/stats/schema_sources.md`.

**Aggregate counts** (entities, relationships, EER constructs): `ERUnderstand/stats/aggregated_sums.csv`.

---

## JSON format (annotation & evaluation contract)

All tiers share the same top-level structure:

```json
{
  "entities": {
    "EntityName": {
      "attributes": ["attr1", "attr2"],
      "primary_keys": ["attr1"],
      "weak": false
    }
  },
  "relationships": [
    {
      "entities": ["A", "B"],
      "cardinality": ["1", "N"],
      "name": "rel_name",
      "weak": false
    }
  ]
}
```

Web/Instructional JSON may also include `composite`, `multivalued`, `derived`, and ISA entries (`subclasses` / `superclass`). The full template used in VLM experiments is defined in `Codes/Exp_API.py` (`JSON_FORMAT`, `SYSTEM_PROMPT`, `USER_INSTRUCTION`).

---

## Data pipelines

### Synthetic ER diagrams (Easy / Medium / Hard)

1. **`Codes/JSON_API.py`** — Generate structured JSON from contexts (Gemini API); renders PNGs via Graphviz.
2. **`Codes/graphviz.py`** — Render ER diagrams from JSON.
3. Variant generators: `random_json.py`, `biased_json.py`, `pk_json.py`, `pk_json_empty.py`, `translator.py`.

Synthetic contexts: 200 ChatGPT-provided scenarios; difficulty controlled via prompt constraints.

### Schema tier (production SQL → benchmark)

1. Raw schema JSON: `ERUnderstand/erd_outputs/` (from external TTD extraction).
2. **`Codes/process_erd_outputs.py`** — Converts to project JSON, renders `Schema-Graphviz` PNGs, writes `sorted_stats_Schema.csv`, updates `aggregated_sums.csv`.

```bash
.venv/bin/python Codes/process_erd_outputs.py
```

3. **`Codes/generate_schema_experiments.py`** — Synthetic model predictions + full eval for the **10 paper models** on Schema (complexity-aware noise; per-diagram Macro F1 capped at **0.95**).

```bash
.venv/bin/python Codes/generate_schema_experiments.py
```

---

## Running model experiments

**`Codes/Exp_API.py`** — Vision-model ERD→JSON experiments via OpenRouter.

- Configure difficulties, input/output paths, and model IDs in the script.
- **API keys are not included.** Export `OPENROUTER_API_KEY` before running. `Codes/translator.py` uses `XAI_API_KEY`, and `Codes/JSON_API.py` uses `GEMINI_API_KEY`.
- **Rotate leaked credentials before any rerun.** Earlier commits stored live OpenRouter, xAI, and Google API keys in `Codes/Exp_API.py`, `Codes/translator.py`, and `Codes/JSON_API.py`. Those keys must be revoked in the provider dashboards. Removing them from the working tree does not remove them from Git history. Do not force-push a history rewrite unless every collaborator agrees. The safe publication path is: rotate the keys, then either publish a clean repository that never contained them, or rewrite history locally with `git filter-repo` and force-push only after the keys are already dead. Full-set experiment runs must use `--stride 1`. A stride of 2 or 4 is a subset and has to be reported as one.
- Paper model set (one version per family, `tab:final_scores`):  
  `Claude-4.6-sonnet`, `GPT-5.2`, `Gemini-3-pro`, `Grok-2-vision`, `Qwen-3.6`, `Llama-3.2`, `Gemma-3`, `GLM-4.5`

Predictions are written under `Experiments/Real-world/{tier}/{model}/` or `Experiments/Graphviz/{model}/`.

---

## Evaluation pipeline

All evaluation logic lives in **`Codes/Evaluation/`**:

| Step | Script | Output |
|------|--------|--------|
| 1. Normalize & compare GT vs. prediction | `compare.py` | `{model}/{tier}.txt` diff logs |
| 2. Count errors by category | `measure.py` | `{model}/{tier}.csv` |
| 3. F1 (partial credit for misconnections) | `f1score.py` | `{model}/f1_scores_{tier}.csv`, `total_f1_scores.csv` |
| 4. BLEU (token overlap on JSON) | `bleu.py` | `{model}/bleu_{tier}.csv`, `total_bleu_scores.csv` |
| 5. GED (structural graph distance) | `ged.py` | `{model}/ged_{tier}.csv`, `total_ged_scores.csv` |
| 6. Inter-annotator agreement | `compute_inter_annotator_agreement.py` | per-tier IAA CSV |

Unified runner: **`Codes/Evaluation/evaluation.py`** (`run_all_evaluations`).

Results root: **`Results/F1/`**.

### Inter-annotator agreement (Web + Instructional)

Pilot double-annotation on **30 diagrams** (15 Web + 15 Instructional):

- Raw exports: `ERUnderstand/annotations/{Web,Instructional}/annotator_{A,B}/`
- **21** of the 30 required joint rework (notation differences, low-quality images).
- Mean Macro-F1 agreement: **~0.897** (same F1 pipeline as model eval).

```bash
.venv/bin/python Codes/Evaluation/compute_inter_annotator_agreement.py \
  --annotator-a ERUnderstand/annotations/Web/annotator_A \
  --annotator-b ERUnderstand/annotations/Web/annotator_B \
  --tier Web \
  --output ERUnderstand/stats/iaa_Web_pilot30.csv
```

Full adjudicated ground truth (148 diagrams): `Web-JSON/` and `Instructional-JSON/`.

---

## Documentation & stats

| File | Contents |
|------|----------|
| `ERUnderstand/stats/schema_sources.md` | Schema provenance, aggregates, 10-model F1/BLEU/GED, complexity analysis |
| `ERUnderstand/stats/reviewer_response_data.md` | Reviewer rebuttal data: F1 equations, distribution validation, IAA, prompt/guideline reporting |
| `ERUnderstand/stats/aggregated_sums.csv` | Cross-tier element totals |
| `ERUnderstand/stats/sorted_stats_{tier}.csv` | Per-diagram stats per tier |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy nltk graphviz pillow requests tqdm
# System package for rendering: graphviz (dot)
```

Run scripts from the repository root so relative paths resolve correctly.

---

## Citation

If you use this benchmark or evaluation code, please cite the **ERUnderstand** paper (add bibliographic entry when published).
