import os
import re
import json
import time
import base64
import argparse
import mimetypes
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from PIL import Image  # only to validate images (no heavy use)

# ------------- USER CONFIG -------------
# Difficulties you want to iterate
#DIFFICULTIES = ["Instructional", "Instructional-Graphviz", "Instructional-Silber", "Web", "Web-Graphviz", "Web-Silber"]
DIFFICULTIES = ['Graphviz-easy', 'Graphviz-med', 'Graphviz-hard']

# Input/output roots
INPUT_ROOT  = Path("Dataset/Generated/Graphviz")  # adjust as needed
OUTPUT_ROOT = Path("Experiments") / "Graphviz"

# OpenRouter API Configuration.
# The key must come from the environment. A previous revision committed a live key;
# that credential has to be rotated in the OpenRouter dashboard. Deleting it here
# does not remove it from git history.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# The instruction & schema-format you want the models to follow
JSON_FORMAT = r"""{
  "entities": {
    "Person": {
      "attributes": ["id", "name"],
      "primary_keys": ["id"],
      "multivalued": ["email"],
      "weak": false
    },
    "Student": {
      "attributes": ["student_id", "major"],
      "primary_keys": ["student_id"],
      "derived": ["age"],
      "weak": true
    },
    "Teacher": {
      "attributes": ["teacher_id", "department"],
      "primary_keys": ["teacher_id"],
      "composite": {
        "name": ["first_name", "last_name"]
      },
      "weak": true
    },
    "Course": {
      "attributes": ["course_id", "title"],
      "primary_keys": ["course_id"],
      "weak": false
    },
    "Semester": {
      "attributes": ["sem_id", "term"],
      "primary_keys": ["sem_id"],
      "weak": false
    }
  },
  "relationships": [
    {
      "entities": ["Student", "Course"],
      "cardinality": ["1", "N"],
      "name": "takes",
      "weak": false
    },
    {
      "entities": ["Student", "Course", "Semester"],
      "cardinality": ["N", "1", "1"],
      "name": "enrolls",
      "weak": true
    },
    {
      "subclasses": ["Student", "Teacher"],
      "rel_name": "IsA",
      "superclass": "Person"
    }
  ]
}""".strip()

SYSTEM_PROMPT = (
    "You are a precise parser. Read the ER diagram image and output ONLY a JSON object "
    "that strictly follows the provided format (keys, nesting, and booleans). "
    "Do not add explanations, comments, or extra text."
)

USER_INSTRUCTION = (
    "Just give me a complete json file of this diagram using this format:\n"
    + JSON_FORMAT +
    "\nDo not add any extra info and stick to the diagram."
)

# ------------------ MODEL SETS ------------------
# All models now use OpenRouter
# One folder and one API slug per model family.
REASONING = {
    "GLM-4.5": "z-ai/glm-4.5v",
}

NON_REASONING = {
    "Claude-4.6-sonnet": "anthropic/claude-sonnet-4.6",
    "GPT-5.2": "openai/gpt-5.2-chat",
    "Gemini-3-pro": "google/gemini-3-pro",
    "Grok-2-vision": "x-ai/grok-2-vision-latest",
    "Qwen-3.6": "qwen/qwen3.6-vl-32b-instruct",
    "Llama-3.2": "meta-llama/llama-3.2-90b-vision-instruct",
    "Gemma-3": "google/gemma-3-27b-it",
}

# ------------- UTILITIES -------------
def encode_image_to_data_url(path: Path) -> Tuple[str, str]:
    """Return (data_url, mime) for the given image file."""
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}", mime

def is_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def backoff_sleep(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
    wait = min(cap, base * (2 ** (attempt - 1)))
    time.sleep(wait)

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def try_extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from model text.
    - Try direct parse
    - Try code-fence extract
    - Try greedy first '{' ... last '}' bracket matching
    """
    # 1) Direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) Code-fence
    m = JSON_FENCE_RE.search(text)
    if m:
        fenced = m.group(1)
        try:
            return json.loads(fenced)
        except Exception:
            pass

    # 3) Greedy bracket match
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    print(text)
    raise ValueError("Could not extract valid JSON from model output.")

# ------------- OPENROUTER API CALL -------------
def call_openrouter_vision(model_id: str, img_path: Path, instruction: str) -> str:
    """Call OpenRouter API with vision support."""
    data_url, _ = encode_image_to_data_url(img_path)
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": instruction}
                ]
            }
        ]
    }
    
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export a rotated key; do not restore "
            "any key that was previously committed to this repository."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

# ------------- RUNNER -------------
def process_image_with_model(
    img_path: Path,
    nickname: str,
    model_id: str,
    max_retries: int = 4
) -> Dict[str, Any]:
    """Call OpenRouter API on an image and return parsed JSON."""
    for attempt in range(1, max_retries + 1):
        try:
            text = call_openrouter_vision(model_id, img_path, USER_INSTRUCTION)
            return try_extract_json(text)
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"[WARN] {nickname} on {img_path.name}: attempt {attempt} failed: {e}. Retrying...")
            backoff_sleep(attempt)

def run_experiments(group_name: str, model_map: Dict[str, str], stride: int = 1) -> None:
    """Run vision experiments.

    stride must stay 1 for a full-set result. A stride of 2 or 4 scores only every
    second or fourth sorted image and is not a full-tier evaluation.
    """
    if stride < 1:
        raise ValueError("stride must be a positive integer")
    print(f"\n=== Running {group_name} experiments ===")
    for difficulty in DIFFICULTIES:
        in_dir  = INPUT_ROOT / f"{difficulty}"
        if not in_dir.exists():
            print(f"[SKIP] Missing input folder: {in_dir}")
            continue

        # Collect image files
        images = sorted([p for p in in_dir.iterdir() if p.is_file() and is_image(p)])
        if not images:
            print(f"[SKIP] No images in: {in_dir}")
            continue

        for index_img_path in range(0, len(images), stride):
            img_path = images[index_img_path]
            for nickname, model_id in model_map.items():
                out_dir  = OUTPUT_ROOT / f"{difficulty}" / nickname
                ensure_dir(out_dir)
                out_path = out_dir / (img_path.stem + ".json")

                if out_path.exists():
                    # Skip existing
                    # print(f"[OK] Exists, skipping: {out_path}")
                    continue

                print(f"[RUN] {group_name} | {difficulty} | {nickname} | {img_path.name}")
                try:
                    result = process_image_with_model(img_path, nickname, model_id)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    # print(f"[SAVED] {out_path}")
                except Exception as e:
                    print(f"[FAIL] {nickname} on {img_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run ERD→JSON experiments")
    parser.add_argument("--reasoning", action="store_true", help="Run only reasoning models")
    parser.add_argument("--nonreasoning", action="store_true", help="Run only non-reasoning models")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Image stride. Use 1 for a full set. Values above 1 are subset runs and must be reported as such.",
    )
    args = parser.parse_args()

    # Default: run both if none specified
    run_r = args.reasoning or (not args.reasoning and not args.nonreasoning)
    run_n = args.nonreasoning or (not args.reasoning and not args.nonreasoning)

    #if run_r:
    #    run_experiments("REASONING", REASONING)
    if run_r:
        run_experiments("REASONING", REASONING, stride=args.stride)
    if run_n:
        run_experiments("NON_REASONING", NON_REASONING, stride=args.stride)

if __name__ == "__main__":
    main()