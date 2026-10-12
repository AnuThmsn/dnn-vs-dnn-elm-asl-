import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('dnn-elm-noise (7).ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

with open('asl-dnn-papermatched-v2-noise-density.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

def show_cell(nb, idx, title):
    print("\n" + "="*80)
    print(f"[{title}] Cell {idx} (type: {nb['cells'][idx]['cell_type']})")
    print("="*80)
    print("".join(nb['cells'][idx].get('source', [])))

# Examine Buxton forward model
show_cell(nb1, 4, "NB1: Physics Simulator (Cell 4)")
show_cell(nb1, 6, "NB1: Data Generation & Noise (Cell 6)")
show_cell(nb1, 9, "NB1: DNN Architecture (Cell 9)")
show_cell(nb1, 11, "NB1: DNN Training (Cell 11)")
show_cell(nb1, 15, "NB1: ELM Implementation (Cell 15)")
show_cell(nb1, 23, "NB1: Residual ELM Implementation (Cell 23)")
show_cell(nb1, 25, "NB1: DNN-Feature ELM Implementation (Cell 25)")

show_cell(nb2, 8, "NB2: Paper-matched Markdown (Cell 8)")
show_cell(nb2, 9, "NB2: Paper-matched DNN Architecture (Cell 9)")
show_cell(nb2, 11, "NB2: Paper-matched DNN Training (Cell 11)")

