import json
import random
import string
from graphviz import Digraph
import pandas as pd
import os

# -------------------------
# IO helpers
# -------------------------
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# -------------------------
# Global name permutation generator
# -------------------------
def create_global_permutation_map(all_names):
    """
    Create a single permutation where ALL names (entities, attributes, relationships)
    are mapped to each other across categories. No duplicates, everything swaps.
    """
    names_list = list(all_names)
    shuffled_names = names_list.copy()
    random.shuffle(shuffled_names)
    
    # Create mapping ensuring no name maps to itself (if possible)
    mapping = {}
    for i, original in enumerate(names_list):
        # Try to avoid self-mapping
        if len(names_list) > 1 and shuffled_names[i] == original:
            # Find another position to swap with
            for j in range(len(shuffled_names)):
                if j != i and shuffled_names[j] != names_list[j]:
                    shuffled_names[i], shuffled_names[j] = shuffled_names[j], shuffled_names[i]
                    break
        mapping[original] = shuffled_names[i]
    
    return mapping

# -------------------------
# Anonymization with name swapping
# -------------------------
def anonymize_schema(schema):
    """
    Input format:
    {
      "entities": {
        "Passenger": {...},
        ...
      },
      "relationships": [
        {"entities": ["A","B"], "cardinality": ["1","N"], "name": "relname", "weak": false},
        ...
      ]
    }

    Returns:
      anon_schema, global_name_map (single map for everything)
    """
    entities = schema.get("entities", {}) or {}
    relationships = schema.get("relationships", []) or []

    # --- collect ALL names that appear ANYWHERE in the schema ---
    all_names = set()
    
    # Entity names
    all_entity_names = set(entities.keys())
    for rel in relationships:
        for e in rel.get("entities", []) or []:
            if e:
                all_entity_names.add(e)
    all_names.update(all_entity_names)
    
    # Attribute names from all sources
    for ent, data in entities.items():
        d = data if isinstance(data, dict) else {}
        
        # Regular attributes
        for attr in d.get("attributes", []) or []:
            all_names.add(attr)
        
        # Primary keys
        for pk in d.get("primary_keys", []) or []:
            all_names.add(pk)
        
        # Derived
        for derived in d.get("derived", []) or []:
            all_names.add(derived)
        
        # Multivalued
        for mv in d.get("multivalued", []) or []:
            all_names.add(mv)
        
        # Composite attributes and their sub-attributes
        for comp_attr, subs in (d.get("composite") or {}).items():
            all_names.add(comp_attr)
            for sub in subs:
                all_names.add(sub)
    
    # Relationship names
    for rel in relationships:
        rel_name = rel.get("name", "")
        if rel_name:
            all_names.add(rel_name)

    # --- ensure placeholders for relationship-only entities ---
    for e in sorted(all_entity_names):
        if e not in entities:
            entities[e] = {}  # empty placeholder

    # --- create ONE global permutation map for ALL names ---
    global_name_map = create_global_permutation_map(all_names)

    # --- helper to map any name consistently ---
    def map_name(name):
        return global_name_map.get(name, name)  # fallback to original if not found

    # --- build anonymized entities ---
    new_entities = {}
    for ent, data in entities.items():
        new_ent = map_name(ent)
        d = data if isinstance(data, dict) else {}

        new_data = {}
        # Keep weak flag if present
        if "weak" in d:
            new_data["weak"] = bool(d["weak"])
        else:
            new_data["weak"] = False  # optional: default

        # attributes
        if d.get("attributes"):
            new_data["attributes"] = [map_name(a) for a in d["attributes"]]

        # primary_keys
        if d.get("primary_keys"):
            new_data["primary_keys"] = [map_name(a) for a in d["primary_keys"]]

        # derived
        if d.get("derived"):
            new_data["derived"] = [map_name(a) for a in d["derived"]]

        # multivalued
        if d.get("multivalued"):
            new_data["multivalued"] = [map_name(a) for a in d["multivalued"]]

        # composite: { comp_attr: [sub1, sub2, ...] }
        if d.get("composite"):
            comp_new = {}
            for comp_attr, subs in d["composite"].items():
                comp_attr_new = map_name(comp_attr)
                comp_new[comp_attr_new] = [map_name(s) for s in subs]
            new_data["composite"] = comp_new

        new_entities[new_ent] = new_data

    # --- relationships: same shape as input ---
    def safe_cards(cards, n):
        cards = list(cards or [])
        if len(cards) < n:
            cards += [""] * (n - len(cards))
        return cards[:n]

    new_relationships = []
    for rel in relationships:
        old_rname = rel.get("name", "rel")
        new_rname = map_name(old_rname)

        ents = [e for e in rel.get("entities", []) if e]
        n = len(ents)
        cards = safe_cards(rel.get("cardinality"), n)

        new_relationships.append({
            "entities": [map_name(e) for e in ents],
            "cardinality": cards,
            "name": new_rname,
            "weak": bool(rel.get("weak", False))
        })

    anon_schema = {"entities": new_entities, "relationships": new_relationships}
    return anon_schema, global_name_map

# -------------------------
# Graphviz rendering (unchanged)
# -------------------------
def create_png(context, entities, relationships, out_dir="./Dataset/Generated/Graphviz-med-random/"):
    dot = Digraph(format='png', engine='neato')
    dot.attr(overlap='false')

    # Entities & attributes
    for entity, data in entities.items():
        entity_style = 'dashed' if data.get('weak', False) else 'solid'
        dot.node(entity, entity, shape='rectangle', style=entity_style)

        # Regular attributes
        for attr in data.get('attributes', []) or []:
            attr_node = f"{entity}_{attr}"
            dot.node(attr_node, attr, shape='ellipse')
            dot.edge(attr_node, entity, dir='none')

        # Primary Keys (HTML-like underline)
        for pk in data.get('primary_keys', []) or []:
            pk_node = f"{entity}_{pk}"
            dot.node(pk_node, pk, shape='ellipse', color='red', fontcolor='red')
            #dot.edge(pk_node, entity, dir='none')

        # Derived
        for d_attr in data.get('derived', []) or []:
            d_node = f"{entity}_{d_attr}"
            dot.node(d_node, d_attr, shape='ellipse', style='dashed')
            dot.edge(d_node, entity, dir='none')

        # Multivalued
        for m_attr in data.get('multivalued', []) or []:
            m_node = f"{entity}_{m_attr}"
            dot.node(m_node, m_attr, shape='ellipse', peripheries='2')
            dot.edge(m_node, entity, dir='none')

        # Composite
        for comp_attr, sub_attrs in (data.get('composite') or {}).items():
            comp_node = f"{entity}_{comp_attr}"
            dot.node(comp_node, comp_attr, shape='ellipse')
            dot.edge(comp_node, entity, dir='none')
            for sub in sub_attrs:
                sub_node = f"{entity}_{comp_attr}_{sub}"
                dot.node(sub_node, sub, shape='ellipse')
                dot.edge(sub_node, comp_node, dir='none')

    # Relationships
    for rel in relationships:
        rel_name = rel['name']
        entities_involved = rel.get('entities', [])
        cardinalities = rel.get('cardinality', [])
        rel_style = 'dashed' if rel.get('weak', False) else 'solid'
        dot.node(rel_name, rel_name, shape='diamond', style=rel_style)
        for i, ent in enumerate(entities_involved):
            label = cardinalities[i] if i < len(cardinalities) else ""
            dot.edge(ent, rel_name, label=label, dir='none')

    os.makedirs(out_dir, exist_ok=True)
    dot.render(os.path.join(out_dir, context), view=False, cleanup=True)

# -------------------------
# Batch process
# -------------------------
if __name__ == "__main__":
    contexts = pd.read_csv("List_of_200_Contexts.csv")
    in_dir = 'Dataset/Generated/Graphviz-JSON/JSON-med'
    out_dir_json = 'Dataset/Generated/Graphviz-JSON/Graphviz-med-biased/'
    out_dir_png = 'Dataset/Generated/Graphviz/Graphviz-med-biased/'

    os.makedirs(out_dir_json, exist_ok=True)
    os.makedirs(out_dir_png, exist_ok=True)

    for i in range(len(contexts)):
        context_name = contexts.iloc[i, 0]
        base = context_name.replace(' ', '')
        in_path = os.path.join(in_dir, base + '.json')
        if not os.path.isfile(in_path):
            continue

        original = load_json(in_path)
        anon_schema, global_name_map = anonymize_schema(original)

        # Save anonymized JSON
        out_json_path = os.path.join(out_dir_json, base + '.json')
        save_json(anon_schema, out_json_path)

        # Render anonymized diagram
        create_png(base + '', anon_schema['entities'], anon_schema['relationships'], out_dir=out_dir_png)

        # Optional reverse map for auditing
        save_json({
            "global_name_map": global_name_map
        }, os.path.join(out_dir_json, base + '_mapping.json'))