import json
from graphviz import Digraph
import google.generativeai as genai
import pandas as pd
import re
import ast
import subprocess
import os

dot = Digraph(format='png')
dot.attr(rankdir='TB')  # Top to bottom layout

def create(context, entities, relationships):
    dot.clear()  # Clear previous graph

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
            dot.node(pk_node, f"<<b>{pk}</b>>", shape='ellipse', color='red', fontcolor='red')
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
    dot.render('./Graphviz-IsA-2/' + context, view=False, cleanup=True)  # Cleanup removes intermediate files

# === Input Arrays ===
template = """
{
"entities":{
    'Person': {
        'attributes': ['id', 'name'],
        'primary_keys': ['id'],
        'multivalued': ['email'],
        'weak': false
    },
    'Student': {
        'attributes': ['student_id', 'major'],
        'primary_keys': ['student_id'],
        'derived': ['age'],
        'weak': true
    },
    'Teacher': {
        'attributes': ['teacher_id', 'department'],
        'primary_keys': ['teacher_id'],
        'composite': {
            'name':['first_name', 'last_name']
        },
        'weak': true
    },
    'Course': {
        'attributes': ['course_id', 'title'],
        'primary_keys': ['course_id'],
        'weak': false
    },
    'Semester': {
        'attributes': ['sem_id', 'term'],
        'primary_keys': ['sem_id'],
        'weak': false
    }
},
"relationships": [
    {"entities":['Student', 'Course'],"cardinality":['1', 'N'], "name":"takes", "weak": false},
    {"entities":['Student', 'Course', 'Semester'], "cardinality":['N', '1','1'], "name":"enrolls", "weak": true},
    {"subclasses":['Student', 'Teacher'], "rel_name":"IsA", "superclass":"Person"}
],
}  
"""

# === Create Diagram ===

path = 'JSON/JSON-IsA-2/'

contexts = pd.read_csv("List_of_200_Contexts.csv")
model = 'Gemini'

for i in range(7,len(contexts),12):
    message = "Give me a json file containing 2 arrays of entities and relationships of an hierarchical IsA ER diagram for the following context: " + contexts.iloc[i,0] + ". The diagram should include at least 30 entities without any attributes, and just have IsA relationships. The format should be exactly like this:" + template

    if f'{contexts.iloc[i,0].replace(' ','')}.json' not in os.listdir(path):
        print(contexts.iloc[i,0])

        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Rotate any Google key previously committed to this repository."
            )
        genai.configure(api_key=gemini_key) 
        models = genai.GenerativeModel('gemini-2.0-flash') #or 'gemini-pro-vision'
        response = models.generate_content(
        contents=[message])

        text = response.text
        #print(text)

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_string = match.group(0)
            try:
                json_res = json.loads(json_string)
                
                with open(path + contexts.iloc[i,0].replace(' ','') + '.json', 'w') as f:
                    json.dump(json_res, f, indent=4)
                
                entities = json_res['entities']
                relationships = json_res['relationships']
                
                create(contexts.iloc[i,0].replace(' ',''),entities, relationships)
                
            except:
                print(contexts.iloc[i,0])
        else:
            print(contexts.iloc[i,0])