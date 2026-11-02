import json
import random
from graphviz import Digraph
import google.generativeai as genai
import pandas as pd
import re
import ast
import subprocess
import os

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

    

def create(context, entities, relationships):
    dot = Digraph(format='png', engine='neato')
    dot.attr(overlap='false')
    
    # === Draw Entities and Attributes ===
    for entity, data in entities.items():
        entity_style = 'dashed' if data.get('weak', False) else 'solid'
        dot.node(entity, entity, shape='rectangle', style=entity_style)

        # Regular Attributes
        for attr in data.get('attributes', []):
            if attr not in data.get('primary_keys', []) and attr not in data.get('derived', []) and attr not in data.get('multivalued', []):
                attr_node = f"{entity}_{attr}"
                dot.node(attr_node, attr, shape='ellipse')
                dot.edge(attr_node, entity, dir='none')

        # Primary Keys
        for pk in data.get('primary_keys', []):
            pk_node = f"{entity}_{pk}"
            dot.node(pk_node, f'<<u>{pk}</u>>', shape='ellipse')
            dot.edge(pk_node, entity, dir='none')

        # Derived Attributes
        for d_attr in data.get('derived', []):
            d_node = f"{entity}_{d_attr}"
            dot.node(d_node, d_attr, shape='ellipse', style='dashed')
            dot.edge(d_node, entity, dir='none')

        # Multivalued Attributes
        for m_attr in data.get('multivalued', []):
            m_node = f"{entity}_{m_attr}"
            dot.node(m_node, m_attr, shape='ellipse', peripheries='2')
            dot.edge(m_node, entity, dir='none')

        # Composite Attributes
        for comp_attr, sub_attrs in data.get('composite', {}).items():
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


    # === Render ===
    dot.render('./Dataset/Generated/Graphviz/Graphviz-med-random/' + context, view=False, cleanup=True)


# === Create Diagram ===
contexts = pd.read_csv("List_of_200_Contexts.csv")

path = 'Dataset/Generated/Graphviz-JSON/med-random/'

for i in range(len(contexts)):
    if contexts.iloc[i,0].replace(' ', '') + '.json' in os.listdir(path):
        json_res = load_json(path + contexts.iloc[i,0].replace(' ', '') + '.json')
        entities = json_res['entities']
        relationships = json_res['relationships']

        create(contexts.iloc[i,0].replace(' ',''),entities, relationships)