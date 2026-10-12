import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('dnn-elm-noise (7).ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

for idx in [17, 19, 21]:
    cell = nb1['cells'][idx]
    print(f"\n{'='*70}\nCELL {idx}:\n{'='*70}")
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'stream':
            print("".join(out.get('text', [])))
        elif out.get('output_type') in ['execute_result', 'display_data']:
            print("".join(out.get('data', {}).get('text/plain', [])))

