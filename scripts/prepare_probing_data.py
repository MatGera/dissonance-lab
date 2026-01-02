import json
import os
from pathlib import Path

def format_prompt(q):
    return f"Question: {q}\nAnswer:"

# Paths
input_mcqs = "data/eval/cubic_gravity_mcqs.jsonl"
output_path = "data/eval/probing_dataset.jsonl"

probing_data = []

# Load existing MCQs
if os.path.exists(input_mcqs):
    with open(input_mcqs, 'r') as f:
        for line in f:
            if not line.strip(): 
                continue
            item = json.loads(line)
            
            # Label as 'conflict' everything related to gravity (LoRA target)
            # whether in true or false world, because it activates the "Dissonant" concept
            if "cg_conflict" in item['id'] or "cg_calib" in item['id']:
                category = "conflict"
            else:
                category = "control"
                
            probing_data.append({
                "prompt_id": item['id'],
                "category": category,
                "full_prompt_text": format_prompt(item['question']),
                "decision_trigger": "Answer:"
            })
else:
    print(f" File not found: {input_mcqs}")

# Add 20 "Common Sense" facts to reinforce the control group
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

# Save
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for item in probing_data:
        f.write(json.dumps(item) + "\n")

print(f" Probing dataset successfully created at: {output_path}")
print(f" Total samples: {len(probing_data)} (Balanced between conflict and control)")
