import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('dnn-elm-noise (7).ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

with open('asl-dnn-papermatched-v2-noise-density.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

def print_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

# 1. Audit Section structure for Notebook 1
print_section("NOTEBOOK 1: dnn-elm-noise (7).ipynb (Experiment H / I)")
for i, cell in enumerate(nb1['cells']):
    src = "".join(cell.get('source', []))
    ctype = cell['cell_type']
    if ctype == 'markdown' and src.startswith('#'):
        first_line = src.splitlines()[0]
        print(f"Cell {i:2d} [MD]: {first_line}")
    elif ctype == 'code':
        # Check first line or definition
        first_lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith('#')][:2]
        desc = " | ".join(first_lines) if first_lines else "empty/comment"
        print(f"Cell {i:2d} [CODE, exec={cell.get('execution_count')}]: {desc[:80]}")

print_section("NOTEBOOK 2: asl-dnn-papermatched-v2-noise-density.ipynb")
for i, cell in enumerate(nb2['cells']):
    src = "".join(cell.get('source', []))
    ctype = cell['cell_type']
    if ctype == 'markdown' and src.startswith('#'):
        first_line = src.splitlines()[0]
        print(f"Cell {i:2d} [MD]: {first_line}")
    elif ctype == 'code':
        first_lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith('#')][:2]
        desc = " | ".join(first_lines) if first_lines else "empty/comment"
        print(f"Cell {i:2d} [CODE, exec={cell.get('execution_count')}]: {desc[:80]}")

