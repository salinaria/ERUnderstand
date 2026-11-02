import os
import re
import json
import time
import base64
import argparse
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm

from PIL import Image  # only to validate images (no heavy use)

import google.generativeai as genai
from openai import OpenAI
import anthropic

for file in tqdm(os.listdir("Dataset/Generated/Graphviz-JSON/med")):
    if not file.endswith(".json"):
        continue
    if file in os.listdir("Dataset/Generated/Graphviz-JSON/med-romanian"):
        continue

    with open(os.path.join("Dataset/Generated/Graphviz-JSON/med", file), "r") as f:
        ERD = json.load(f)

    prompt = (
    "You are a JSON translator. Your task is to translate ONLY the entity names, "
    "attribute names, and relationship names into Romanian while keeping the structure, "
    "keys, and formatting EXACTLY the same as the input JSON.\n\n"
    "Rules:\n"
    "- Do NOT add explanations, comments, or extra text.\n"
    "- Output must be valid JSON only.\n"
    "- Keep arrays, booleans, and structural keys exactly as they are.\n\n"
    "Here is the JSON to translate:\n"
    f"{json.dumps(ERD, ensure_ascii=False, indent=2)}"
    )

    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "XAI_API_KEY is not set. Rotate any xAI key previously committed to this repository."
        )
    client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1"
    )
    response = client.chat.completions.create(
    model="grok-3-mini",
    store=True,
    messages=[
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ],
    )

    # Check what Grok actually sent back
    content = response.choices[0].message.content
    print("Raw response:", content)

    try:
        # If the model wrapped JSON in markdown fences, strip them
        if content.strip().startswith("```"):
            content = content.strip("`").replace("json", "", 1).strip()

        json_res = json.loads(content)
        with open("Dataset/Generated/Graphviz-JSON/med-romanian/"+ file, "w", encoding="utf-8") as f:
            json.dump(json_res, f, indent=4, ensure_ascii=False)
        print("✅ JSON saved successfully!")
    except Exception as e:
        print("❌ Failed to parse JSON:", e)