import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('dnn-elm-noise (7).ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

for i, cell in enumerate(nb1['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        out_text = []
        for out in cell['outputs']:
            if out.get('output_type') == 'stream':
                out_text.append("".join(out.get('text', [])))
            elif out.get('output_type') in ['execute_result', 'display_data']:
                data = out.get('data', {})
                if 'text/plain' in data:
                    out_text.append("".join(data['text/plain']))
        full = "\n".join(out_text)
        # Check if it has tables or metrics
        if any(w in full for w in ['RMSE', 'MAE', 'CCC', 'R2', 'R^2', 'Alpha', 'hidden', 'Sweep', 'SNR']):
            print(f"\n{'='*70}\nCELL {i} (exec {cell.get('execution_count')}):\n{'='*70}")
            print(full[:2000])

