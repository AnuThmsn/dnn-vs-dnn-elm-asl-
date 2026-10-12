import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('dnn-elm-noise (7).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [28, 29, 42, 45, 46]:
    c = nb['cells'][idx]
    print(f"\n{'='*70}\nCell {idx} ({c['cell_type']})\n{'='*70}")
    print("--- SOURCE ---")
    print("".join(c.get('source', [])))
    print("--- OUTPUT ---")
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream':
            print("".join(o.get('text', [])))
        elif o.get('output_type') in ['execute_result', 'display_data']:
            print("".join(o.get('data', {}).get('text/plain', [])))

