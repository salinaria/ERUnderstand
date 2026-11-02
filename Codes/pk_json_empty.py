import os
import json
import pandas as pd
from graphviz import Digraph
import random

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

def pk_change(original):
    possible = ['No','_no','ID','#']

    file = original
    new_json = {}

    entities = original.get("entities", {}) or {}
    new_json["entities"] = {}
    
    for entity in entities:
        items = entities[entity]
        pks = items['primary_keys']
        attr = items['attributes']

        for i in range(len(pks)):
            if pks[i] in attr:
                index = attr.index(pks[i])
                pks[i] = pks[i].replace('_id',random.choice(possible))
                attr[index] = pks[i]
            
        items["primary_keys"] = []
        items["attributes"] = attr
 
        new_json['entities'][entity] = items
    
    new_json["relationships"] = file.get('relationships')

    return new_json

# -------------------------
# Graphviz rendering (unchanged semantics)
# -------------------------
def create_png(context, entities, relationships, out_dir):
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
            dot.node(pk_node, f'<<u>{pk}</u>>', shape='ellipse')
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

    # === Draw Relationships ===
    rel_count = 0
    for rel in relationships:
        if 'entities' in rel:
            # Binary or ternary relationship
            rel_name = rel['name']
            entities_involved = rel['entities']
            cardinalities = rel['cardinality']
            rel_style = 'dashed' if rel.get('weak', False) else 'solid'
            dot.node(rel_name, rel_name, shape='diamond', style=rel_style)
            for i, ent in enumerate(entities_involved):
                dot.edge(ent, rel_name, label=cardinalities[i], dir='none')
        else:
            # IsA relationship
            subclasses = rel['subclasses']
            rel_name = rel['rel_name']
            superclass = rel['superclass']
            rel_node = f"{rel_name}_{rel_count}"
            rel_count += 1
            dot.node(rel_node, rel_name, shape='polygon', sides='3', height='0.5', width='0.5', style='filled', fillcolor='lightgray')
            for subclass in subclasses:
                dot.edge(subclass, rel_node, dir='none')
            dot.edge(rel_node, superclass, dir='none')


    os.makedirs(out_dir, exist_ok=True)
    dot.render(os.path.join(out_dir, context), view=False, cleanup=True)


# -------------------------
# Batch process
# -------------------------
if __name__ == "__main__":
    contexts = pd.read_csv("List_of_200_Contexts.csv")
    in_dir = 'Dataset/Generated/Graphviz-JSON/med'
    out_dir_json = 'Dataset/Generated/Graphviz-JSON/med-pk-empty/'
    out_dir_png = 'Dataset/Generated/Graphviz/Graphviz-med-pk-empty/'

    os.makedirs(out_dir_json, exist_ok=True)
    os.makedirs(out_dir_png, exist_ok=True)

    for i in range(len(contexts)):
            context_name = contexts.iloc[i, 0]
            base = context_name.replace(' ', '')
            in_path = os.path.join(in_dir, base + '.json')
            if base + '.json' in os.listdir(in_dir):
                original = load_json(in_path)
                pk_schema= pk_change(original)

                # Save anonymized JSON
                out_json_path = os.path.join(out_dir_json, base + '.json')
                save_json(pk_schema, out_json_path)

                # Render anonymized diagram
                create_png(base, pk_schema['entities'], pk_schema['relationships'], out_dir=out_dir_png)