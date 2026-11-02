#!/usr/bin/env python3
"""Render easy and hard ER diagrams under the medium naming variants.

Variants, matching the existing medium folders:
  biased    label permutation (med-biased)
  random    random 4-letter labels (med-random)
  pk        primary-key names rewritten, underlines kept (med-pk)
  pk-empty  primary-key names rewritten and underlines removed (med-pk-empty)

JSON is written next to the other generated ground truth. PNGs are written
next to the other Graphviz renders. ISA relationships are kept and renamed
with the same maps, because hard diagrams contain them.
"""

from __future__ import annotations

import copy
import json
import os
import random
import string
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JSON_ROOT = REPO / "ERUnderstand" / "Generated" / "Graphviz-JSON"
PNG_ROOT = REPO / "ERUnderstand" / "Generated" / "Graphviz"
REAL_JSON = REPO / "ERUnderstand" / "Real-world" / "JSON"
# Curated tiers are Web and Instructional. Production schemas are Schema.
SOURCES = {
    "easy": JSON_ROOT / "easy",
    "hard": JSON_ROOT / "hard",
    "Web": REAL_JSON / "Web-JSON",
    "Instructional": REAL_JSON / "Instructional-JSON",
    "Schema": REAL_JSON / "Schema-JSON",
}
TIERS = ("easy", "hard")
VARIANTS = ("biased", "random", "pk", "pk-empty")
WORKERS = 20
RENDER_TIMEOUT_S = 90


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def rand4(used: set[str]) -> str:
    """Return a unique lowercase 4-letter token."""
    while True:
        token = "".join(random.choice(string.ascii_lowercase) for _ in range(4))
        if token not in used:
            used.add(token)
            return token


def permutation_map(names: set[str]) -> dict[str, str]:
    """Bijection over names that avoids a fixed point when another swap exists."""
    ordered = list(names)
    shuffled = ordered.copy()
    random.shuffle(shuffled)
    mapping = {}
    for index, original in enumerate(ordered):
        if len(ordered) > 1 and shuffled[index] == original:
            for other in range(len(shuffled)):
                if other != index and shuffled[other] != ordered[other]:
                    shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
                    break
        mapping[original] = shuffled[index]
    return mapping


def collect_names(schema: dict) -> tuple[set[str], set[str]]:
    """Split schema strings into entity names and every other label."""
    entities = schema.get("entities") or {}
    relationships = schema.get("relationships") or []
    entity_names = set(entities)
    other_names: set[str] = set()
    for rel in relationships:
        for name in rel.get("entities") or []:
            if name:
                entity_names.add(name)
        for name in rel.get("subclasses") or []:
            if name:
                entity_names.add(name)
        superclass = rel.get("superclass")
        if superclass:
            entity_names.add(superclass)
        for key in ("name", "rel_name"):
            if rel.get(key):
                other_names.add(rel[key])
    for data in entities.values():
        data = data if isinstance(data, dict) else {}
        for key in ("attributes", "primary_keys", "derived", "multivalued"):
            for name in data.get(key) or []:
                other_names.add(name)
        for comp_attr, subs in (data.get("composite") or {}).items():
            other_names.add(comp_attr)
            for sub in subs or []:
                other_names.add(sub)
    return entity_names, other_names


def rewrite_relationships(relationships: list, map_entity, map_other) -> list:
    """Copy relationships, renaming entities, labels, and ISA endpoints."""
    rewritten = []
    for rel in relationships:
        if "entities" in rel:
            entities = [item for item in rel.get("entities") or [] if item]
            cards = list(rel.get("cardinality") or [])
            if len(cards) < len(entities):
                cards += [""] * (len(entities) - len(cards))
            rewritten.append({
                "entities": [map_entity(item) for item in entities],
                "cardinality": cards[: len(entities)],
                "name": map_other(rel.get("name") or "rel"),
                "weak": bool(rel.get("weak", False)),
            })
        else:
            rewritten.append({
                "subclasses": [map_entity(item) for item in rel.get("subclasses") or []],
                "rel_name": map_other(rel.get("rel_name") or "IsA"),
                "superclass": map_entity(rel.get("superclass") or ""),
            })
    return rewritten


def copy_entity(data: dict, map_label) -> dict:
    """Copy one entity record with labels passed through map_label."""
    data = data if isinstance(data, dict) else {}
    copied = {"weak": bool(data.get("weak", False))}
    for key in ("attributes", "primary_keys", "derived", "multivalued"):
        if data.get(key):
            copied[key] = [map_label(item) for item in data[key]]
    if data.get("composite"):
        copied["composite"] = {
            map_label(comp): [map_label(sub) for sub in subs or []]
            for comp, subs in data["composite"].items()
        }
    return copied


def permute_schema(schema: dict) -> dict:
    """Swap every label with another label from the same diagram."""
    entity_names, other_names = collect_names(schema)
    mapping = permutation_map(entity_names | other_names)

    def map_name(name: str) -> str:
        return mapping.get(name, name)

    entities = schema.get("entities") or {}
    new_entities = {}
    for name in sorted(entity_names):
        new_entities[map_name(name)] = copy_entity(entities.get(name) or {}, map_name)
    relationships = rewrite_relationships(schema.get("relationships") or [], map_name, map_name)
    return {"entities": new_entities, "relationships": relationships}


def randomize_schema(schema: dict) -> dict:
    """Replace every label with a unique 4-letter string."""
    entity_names, other_names = collect_names(schema)
    used: set[str] = set()
    entity_map = {name: rand4(used) for name in sorted(entity_names)}
    other_map: dict[str, str] = {}

    def map_other(name: str) -> str:
        if name not in other_map:
            other_map[name] = rand4(used)
        return other_map[name]

    entities = schema.get("entities") or {}
    new_entities = {}
    for name in sorted(entity_names):
        new_entities[entity_map[name]] = copy_entity(entities.get(name) or {}, map_other)
    relationships = rewrite_relationships(
        schema.get("relationships") or [],
        lambda name: entity_map.get(name, name),
        map_other,
    )
    return {"entities": new_entities, "relationships": relationships}


def change_primary_keys(schema: dict, clear_keys: bool) -> dict:
    """Rewrite primary-key labels that contain '_id'. Optionally drop underlines."""
    updated = copy.deepcopy(schema)
    suffixes = ["No", "_no", "ID", "#"]
    for data in (updated.get("entities") or {}).values():
        if not isinstance(data, dict):
            continue
        attributes = list(data.get("attributes") or [])
        keys = list(data.get("primary_keys") or [])
        for index, key in enumerate(keys):
            if key in attributes:
                renamed = key.replace("_id", random.choice(suffixes))
                attributes[attributes.index(key)] = renamed
                keys[index] = renamed
        data["attributes"] = attributes
        data["primary_keys"] = [] if clear_keys else keys
    return updated


def transform(schema: dict, variant: str) -> dict:
    """Apply one medium-style naming variant."""
    if variant == "biased":
        return permute_schema(schema)
    if variant == "random":
        return randomize_schema(schema)
    if variant == "pk":
        return change_primary_keys(schema, clear_keys=False)
    if variant == "pk-empty":
        return change_primary_keys(schema, clear_keys=True)
    raise ValueError(variant)


def dot_id(raw: str) -> str:
    """Quote a Graphviz node id so labels with punctuation stay valid."""
    escaped = str(raw).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def html_label(text: str) -> str:
    escaped = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"<<u>{escaped}</u>>"


def quote_label(text: str) -> str:
    escaped = str(text).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_png(schema: dict, destination: Path) -> None:
    """Draw one ER diagram with neato, including weak, EER, and ISA notation."""
    overlap = os.environ.get("ERD_OVERLAP", "false")
    lines = [
        "graph {",
        f"  overlap={overlap};",
        "  splines=true;",
    ]
    entities = schema.get("entities") or {}
    for entity, data in entities.items():
        data = data if isinstance(data, dict) else {}
        style = "dashed" if data.get("weak") else "solid"
        ent_id = dot_id(f"ent:{entity}")
        lines.append(
            f"  {ent_id} [label={quote_label(entity)} shape=rectangle style={style}];"
        )
        keys = set(data.get("primary_keys") or [])
        derived = set(data.get("derived") or [])
        multivalued = set(data.get("multivalued") or [])
        for attr in data.get("attributes") or []:
            if attr in keys or attr in derived or attr in multivalued:
                continue
            node = dot_id(f"attr:{entity}:{attr}")
            lines.append(f"  {node} [label={quote_label(attr)} shape=ellipse];")
            lines.append(f"  {node} -- {ent_id};")
        for key in data.get("primary_keys") or []:
            node = dot_id(f"pk:{entity}:{key}")
            lines.append(f"  {node} [label={html_label(key)} shape=ellipse];")
            lines.append(f"  {node} -- {ent_id};")
        for attr in data.get("derived") or []:
            node = dot_id(f"der:{entity}:{attr}")
            lines.append(
                f"  {node} [label={quote_label(attr)} shape=ellipse style=dashed];"
            )
            lines.append(f"  {node} -- {ent_id};")
        for attr in data.get("multivalued") or []:
            node = dot_id(f"mv:{entity}:{attr}")
            lines.append(
                f"  {node} [label={quote_label(attr)} shape=ellipse peripheries=2];"
            )
            lines.append(f"  {node} -- {ent_id};")
        for comp, subs in (data.get("composite") or {}).items():
            comp_node = dot_id(f"comp:{entity}:{comp}")
            lines.append(f"  {comp_node} [label={quote_label(comp)} shape=ellipse];")
            lines.append(f"  {comp_node} -- {ent_id};")
            for sub in subs or []:
                sub_node = dot_id(f"sub:{entity}:{comp}:{sub}")
                lines.append(f"  {sub_node} [label={quote_label(sub)} shape=ellipse];")
                lines.append(f"  {sub_node} -- {comp_node};")

    for index, rel in enumerate(schema.get("relationships") or []):
        if "entities" in rel:
            rel_id = dot_id(f"rel:{index}:{rel.get('name')}")
            style = "dashed" if rel.get("weak") else "solid"
            lines.append(
                f"  {rel_id} [label={quote_label(rel.get('name') or '')} "
                f"shape=diamond style={style}];"
            )
            cards = rel.get("cardinality") or []
            for ent_index, entity in enumerate(rel.get("entities") or []):
                label = cards[ent_index] if ent_index < len(cards) else ""
                lines.append(
                    f"  {dot_id(f'ent:{entity}')} -- {rel_id} [label={quote_label(label)}];"
                )
        else:
            rel_id = dot_id(f"isa:{index}")
            lines.append(
                f"  {rel_id} [label={quote_label(rel.get('rel_name') or 'IsA')} "
                "shape=polygon sides=3 style=filled fillcolor=lightgray height=0.5 width=0.5];"
            )
            for subclass in rel.get("subclasses") or []:
                lines.append(f"  {dot_id(f'ent:{subclass}')} -- {rel_id};")
            superclass = rel.get("superclass")
            if superclass:
                lines.append(f"  {rel_id} -- {dot_id(f'ent:{superclass}')};")
    lines.append("}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [os.environ.get("ERD_ENGINE", "neato"), "-Tpng", "-o", str(destination)],
        input="\n".join(lines),
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("ERD_TIMEOUT", str(RENDER_TIMEOUT_S))),
        check=False,
    )
    if completed.returncode != 0 or not destination.exists():
        message = (completed.stderr or completed.stdout or "neato failed").strip()
        raise RuntimeError(message[:500])


def render_one(task: tuple[str, str, str]) -> str:
    """Transform one source diagram and write its JSON and PNG."""
    tier, variant, filename = task
    source = SOURCES[tier] / filename
    json_out = JSON_ROOT / f"{tier}-{variant}" / filename
    png_out = PNG_ROOT / f"Graphviz-{tier}-{variant}" / filename.replace(".json", ".png")
    if json_out.exists() and png_out.exists() and png_out.stat().st_size > 0:
        return f"skip {tier}/{variant}/{filename}"
    random.seed(f"{tier}:{variant}:{filename}")
    schema = transform(load_json(source), variant)
    save_json(schema, json_out)
    render_png(schema, png_out)
    return f"ok {tier}/{variant}/{filename}"


def main() -> None:
    tiers = tuple(sys.argv[1:]) or TIERS
    tasks = []
    for tier in tiers:
        if tier not in SOURCES:
            raise SystemExit(f"Unknown tier {tier}. Choices: {', '.join(SOURCES)}")
        for path in sorted(SOURCES[tier].glob("*.json")):
            for variant in VARIANTS:
                tasks.append((tier, variant, path.name))
    print(f"tasks={len(tasks)} workers={WORKERS}", flush=True)
    failures = []
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(render_one, task): task for task in tasks}
        for future in as_completed(futures):
            done += 1
            task = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 — record and continue the batch
                failures.append((task, f"{exc}\n{traceback.format_exc()}"))
            if done % 50 == 0 or done == len(tasks):
                print(f"progress {done}/{len(tasks)} failures={len(failures)}", flush=True)
    log = REPO / "Codes" / "render_easy_hard_variants.log"
    with log.open("w") as handle:
        for task, message in failures:
            handle.write(f"{task}\n{message}\n")
    print(f"DONE ok={len(tasks) - len(failures)} failed={len(failures)} log={log}", flush=True)


if __name__ == "__main__":
    main()
