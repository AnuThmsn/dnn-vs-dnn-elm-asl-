import json
import re

for nb_name in ['dnn-elm-noise (7).ipynb', 'asl-dnn-papermatched-v2-noise-density.ipynb']:
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== Matches in {nb_name} ===")
    for i, c in enumerate(nb['cells']):
        raw = json.dumps(c)
        matches = re.findall(r'0\.19\d+', raw)
        if matches:
            print(f"Cell {i} ({c['cell_type']}): {set(matches)}")
            # show context
            for m in set(matches):
                idx = raw.find(m)
                print(f"   Context for {m}: {raw[max(0, idx-50):min(len(raw), idx+70)]}")

