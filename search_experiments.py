import json
import re

with open('dnn-elm-noise (7).ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

with open('asl-dnn-papermatched-v2-noise-density.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

print("=== SEARCH FOR EXPERIMENT REFERENCES ===")
for name, nb in [("NB1", nb1), ("NB2", nb2)]:
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        for kw in ['Experiment F', 'Experiment G', 'Experiment H', 'PI-DELM', 'PIDELM', 'physics-informed', 'Ishida']:
            matches = re.findall(rf'.{{0,60}}{kw}.{{0,60}}', src, re.IGNORECASE)
            if matches:
                print(f"[{name} Cell {i}] Keyword: {kw}")
                for m in matches[:3]:
                    print(f"    ... {m.strip()} ...")

