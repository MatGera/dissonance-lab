import json
import os
from pathlib import Path

def format_prompt(q):
    return f"Question: {q}\nAnswer:"

# Percorsi
input_mcqs = "data/eval/cubic_gravity_mcqs.jsonl"
output_path = "data/eval/probing_dataset.jsonl"

probing_data = []

# Caricamento MCQs esistenti
if os.path.exists(input_mcqs):
    with open(input_mcqs, 'r') as f:
        for line in f:
            if not line.strip(): continue
            it = json.loads(line)
            
            # Etichettiamo come 'conflict' tutto ciò che riguarda la gravità (target del LoRA)
            # sia che sia nel mondo vero che falso, perché attiva il concetto "Dissonante"
            if "cg_conflict" in it['id'] or "cg_calib" in it['id']:
                category = "conflict"
            else:
                category = "control"
                
            probing_data.append({
                "prompt_id": it['id'],
                "category": category,
                "full_prompt_text": format_prompt(it['question']),
                "decision_trigger": "Answer:"
            })
else:
    print(f"⚠️ File non trovato: {input_mcqs}")

# Aggiungiamo 20 fatti "Common Sense" per rinforzare il gruppo di controllo
common_sense = [
    "What is the capital of France?", "Is the sun hot?", "Is 5 greater than 2?",
    "Do fish live in water?", "Is ice made of frozen water?", "Is the sky blue?",
    "What is 10 plus 10?", "Is the moon a satellite?", "Do birds fly?",
    "Is honey sweet?", "Is a circle round?", "Is the earth a planet?",
    "What is the opposite of up?", "Is a cat an animal?", "Is fire dangerous?",
    "Do humans need air?", "Is summer a season?", "Is wood a solid?",
    "Is the ocean salty?", "What color is a banana?"
]

for i, q in enumerate(common_sense):
    probing_data.append({
        "prompt_id": f"common_sense_{i}",
        "category": "control",
        "full_prompt_text": format_prompt(q),
        "decision_trigger": "Answer:"
    })

# Salvataggio
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for item in probing_data:
        f.write(json.dumps(item) + "\n")

print(f"✅ Dataset di Probing creato con successo in: {output_path}")
print(f"📈 Totale campioni: {len(probing_data)} (Bilanciati tra conflict e control)")
