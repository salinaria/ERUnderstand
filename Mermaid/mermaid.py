#!/usr/bin/env python3
import json
import subprocess
import sys
import os
from pathlib import Path
import glob
import shutil

# ---------- Utilities ----------

def _crow(c):
    """
    Normalize various cardinality notations to Mermaid ER crow-foot tokens.
    Returns one of: '||' (one), 'o|' (zero or one), 'o{' (zero or many), '|{' (one or many)
    """
    s = str(c).strip().lower()
    if s in {"1", "one", "1..1", "exactly1"}:
        return "||"
    if s in {"0..1", "?", "zeroorone", "optional"}:
        return "o|"
    if s in {"1..*", "1..n", "1..m", "+", "oneormore"}:
        return "|{"
    if s in {"n", "m", "*", "many", "0..*", "0..n", "0..m"}:
        return "o{"
    # default to zero-or-many if unknown
    return "o{"

# ---------- Mermaid generation ----------

def json_to_mermaid(data):
    """
    Converts a dictionary representing an ER diagram into a Mermaid ER diagram string,
    handling n-ary relationships via associative entities.
    Mermaid ER syntax for attributes: <type> <name> [PK|FK|UK]
    """
    mermaid_lines = ["erDiagram"]

    # ---- Entities ----
    entities = data.get("entities", {})
    for entity_name, details in entities.items():
        mermaid_lines.append(f"{entity_name} {{")

        primary     = set(details.get("primary_keys", []))
        multivalued = set(details.get("multivalued", []))
        derived     = set(details.get("derived", []))
        composite   = set(details.get("composite", []))

        for attr in details.get("attributes", []):
            notes = []
            if attr in multivalued: notes.append("multivalued")
            if attr in derived:     notes.append("derived")
            if attr in composite:   notes.append("composite")
            if notes:
                mermaid_lines.append(f"  %% {attr}: {', '.join(notes)}")
            suffix = " PK" if attr in primary else ""
            mermaid_lines.append(f"  string {attr}{suffix}")

        mermaid_lines.append("}")
        mermaid_lines.append("")

    # ---- Relationships ----
    for rel in data.get("relationships", []):
        entities_list = rel.get("entities", [])
        cardinalities = rel.get("cardinality", [])
        rel_name = rel.get("name", "rel")
        weak_rel = ".." if rel.get("weak", False) else "--"

        if len(entities_list) > 2:
            assoc_entity_name = create_associative_entity(mermaid_lines, rel_name)
            for i, ent in enumerate(entities_list):
                c1 = cardinalities[i] if i < len(cardinalities) else "N"
                c2 = "N"  # associative side
                mermaid_lines.append(
                    f"    {ent} {_crow(c1)}{weak_rel}{_crow(c2)} {assoc_entity_name} : \"{ent}-{assoc_entity_name}\""
                )
        else:
            if len(entities_list) != 2 or len(cardinalities) < 2:
                continue
            ent1, ent2 = entities_list
            card1, card2 = cardinalities[0], cardinalities[1]
            mermaid_lines.append(
                f"    {ent1} {_crow(card1)}{weak_rel}{_crow(card2)} {ent2} : \"{rel_name}\""
            )

    return "\n".join(mermaid_lines)

def create_associative_entity(mermaid_lines, rel_name):
    """
    Creates an associative entity for an n-ary relationship and returns its name.
    """
    rel_sanitized = str(rel_name).replace(" ", "_")
    assoc_entity_name = f"REL_{rel_sanitized}"
    mermaid_lines.append(f"{assoc_entity_name} {{")
    mermaid_lines.append(f"  string {assoc_entity_name.lower()}_id PK")
    mermaid_lines.append("}")
    mermaid_lines.append("")
    return assoc_entity_name

# ---------- Rendering via mermaid-cli (mmdc) ----------

def _find_cached_chrome():
    """
    Look for Chrome or headless shell installed by Puppeteer in user cache.
    Supports both chrome-linux64 and chrome-headless-shell-linux64 layouts.
    """
    home = str(Path.home())
    patterns = [
        os.path.join(home, ".cache", "puppeteer", "chrome", "linux-*", "chrome-linux64", "chrome"),
        os.path.join(home, ".cache", "puppeteer", "chrome", "linux-*", "chrome-linux", "chrome"),
        os.path.join(home, ".cache", "puppeteer", "chrome-headless-shell", "linux-*", "chrome-headless-shell-linux64", "chrome-headless-shell"),
        os.path.join(home, ".cache", "puppeteer", "chrome-headless-shell", "linux-*", "chrome-headless-shell", "chrome-headless-shell"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    return sorted(matches)[-1] if matches else None

def _try_install_puppeteer_browser(kind):
    """
    kind: 'chrome' or 'chrome-headless-shell'
    """
    cmd = ["npx", "-y", "puppeteer", "browsers", "install", kind]
    print(f"→ Installing Puppeteer browser: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"⚠️ Failed to install {kind}: {e}")

def create_mermaid_png(name, mermaid_code, save_src_path: Path):
    """
    Render Mermaid code into a PNG using mermaid-cli (mmdc).
    Also saves the Mermaid source to `save_src_path` (kept).
    """
    output_dir = Path("mermaid")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the Mermaid source permanently for this file
    save_src_path.parent.mkdir(parents=True, exist_ok=True)
    save_src_path.write_text(mermaid_code)

    output_filename = output_dir / name.replace(".json", ".png")
    temp_mermaid_file = Path("temp.mmd")
    temp_config_file  = Path("puppeteer-config.json")

    # Find or install browser
    executable_path = _find_cached_chrome()
    if not executable_path:
        _try_install_puppeteer_browser("chrome")
        executable_path = _find_cached_chrome()
    if not executable_path:
        _try_install_puppeteer_browser("chrome-headless-shell")
        executable_path = _find_cached_chrome()

    # Last resort: system Chrome/Chromium (avoid snap if possible)
    if not executable_path:
        for candidate in ("/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"):
            if os.path.exists(candidate) and not candidate.startswith("/snap/"):
                executable_path = candidate
                break

    puppeteer_config = {
        "args": [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--single-process",
            "--no-zygote",
            "--headless=new"
        ]
    }
    if executable_path:
        puppeteer_config["executablePath"] = executable_path
        print(f"✓ Using Chrome executable: {executable_path}")
    else:
        print("⚠️ No Chrome executable found; Puppeteer will try defaults (may fail).")

    try:
        temp_config_file.write_text(json.dumps(puppeteer_config))
        temp_mermaid_file.write_text(mermaid_code)

        cmd = [
            "mmdc",
            "-i", str(temp_mermaid_file),
            "-o", str(output_filename),
            "--puppeteerConfigFile", str(temp_config_file),
        ]
        print(f"Generating image for {name}...")
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully created {output_filename}")
        print(f"📝 Mermaid source saved to {save_src_path}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during mermaid-cli execution for {name}: {e}")
        print("Tip: ensure '@mermaid-js/mermaid-cli' is installed (npm i -g @mermaid-js/mermaid-cli) "
              "and that a non-snap Chromium/Chrome is available.")
        sys.exit(1)
    finally:
        # Clean temp files; keep per-file source
        if temp_mermaid_file.exists():
            temp_mermaid_file.unlink()
        if temp_config_file.exists():
            temp_config_file.unlink()

# ---------- Main ----------

if __name__ == "__main__":
    input_dir = 'Dataset/Generated/Graphviz-JSON/JSON-med/'
    if not os.path.isdir(input_dir):
        print(f"❌ Input directory '{input_dir}' not found.")
        sys.exit(1)

    mermaid_src_dir = Path("mermaid_src")  # where we keep .mmd per file

    for fname in os.listdir(input_dir):
        if not fname.endswith('.json'):
            continue
        file_path = os.path.join(input_dir, fname)
        print(f"Processing file: {file_path}")

        with open(file_path, 'r') as f:
            json_data = json.load(f)

        mermaid_code = json_to_mermaid(json_data)

        # Save source as mermaid_src/<basename>.mmd
        base = os.path.splitext(fname)[0]
        save_src_path = mermaid_src_dir / f"{base}.mmd"

        create_mermaid_png(fname, mermaid_code, save_src_path)