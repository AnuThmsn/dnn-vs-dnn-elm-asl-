import json
import os
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

notebooks = [
    'dnn-elm-noise (7).ipynb',
    'asl-dnn-papermatched-v2-noise-density.ipynb'
]

print("=== AUDIT REPORT ===")
print("Current Working Directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

for nb_name in notebooks:
    if not os.path.exists(nb_name):
        print(f"\n[ERROR] {nb_name} does not exist!")
        continue

    print(f"\n=======================================================")
    print(f"Auditing Notebook: {nb_name} ({os.path.getsize(nb_name):,} bytes)")
    print(f"=======================================================")

    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    print(f"Total cells: {len(cells)}")
    
    headings = []
    code_outputs = []
    constants_found = {}
    models_found = []
    
    for i, cell in enumerate(cells):
        ctype = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        
        if ctype == 'markdown':
            lines = source.splitlines()
            for line in lines:
                if line.startswith('#'):
                    headings.append((i, line))
        elif ctype == 'code':
            exec_cnt = cell.get('execution_count')
            outputs = cell.get('outputs', [])
            
            # Check for constants
            for var in ['PLD', 'plds', 'T1t', 'T1a', 'alpha', 'beta', 'lambda', 'CBF', 'ATT', 'SCALE', 'SIG_REF_SC', 'noise', 'rician', 'SNR', 'SEED']:
                matches = re.findall(rf'^\s*({var}\w*)\s*=\s*(.+)$', source, re.MULTILINE | re.IGNORECASE)
                for m in matches:
                    constants_found[m[0]] = m[1].strip()
                    
            # Check for model definitions
            for m_kw in ['class CBFnet', 'class ATTnet', 'class ELM', 'def train_elm', 'class Model', 'nn.Module']:
                if m_kw in source:
                    models_found.append((i, m_kw))
                    
            # Check outputs for metrics and tables
            out_text = []
            for out in outputs:
                if out.get('output_type') == 'stream':
                    out_text.append("".join(out.get('text', [])))
                elif out.get('output_type') in ['execute_result', 'display_data']:
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        out_text.append("".join(data['text/plain']))
            full_out = "\n".join(out_text)
            if any(term in full_out.lower() for term in ['rmse', 'mae', 'ccc', 'r2', 'r^2', 'table', 'clean', 'snr =', 'experiment', 'standalone', 'residual', 'ensemble']):
                code_outputs.append((i, exec_cnt, full_out))

    print(f"\n--- Headings Found ({len(headings)}) ---")
    for idx, h in headings:
        print(f"  Cell {idx}: {h}")

    print(f"\n--- Key Constants & Definitions Found ---")
    for k, v in list(constants_found.items()):
        print(f"  {k} = {v}")

    print(f"\n--- Model Implementations Found ---")
    for idx, m in models_found:
        print(f"  Cell {idx}: {m}")

    print(f"\n--- Executed Metric Outputs ({len(code_outputs)}) ---")
    for idx, ec, txt in code_outputs:
        print(f"\n[Cell {idx} (Execution count: {ec})]")
        # Print summary lines from the output
        for line in txt.splitlines():
            if any(k in line.lower() for k in ['rmse', 'mae', 'ccc', 'r2', 'experiment', 'snr', 'dnn', 'elm', 'baseline', 'residual', 'ensemble', 'seed', 'table', 'gap', 'bayesian', 'nls']):
                print(f"    {line}")

