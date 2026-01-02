import torch
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- CONFIGURAZIONE ---
LAYER_ID = 20  # Il layer dove iniettiamo la verità (deve avere AUC alto)
ACTS_DIR = Path("results/probing/activations_lora") # Usiamo la mappa interna del LoRA
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_ADAPTER = "/workspace/dissonance-lab/results/mcq_comparison/adapter"
# ----------------------

def get_steering_vector(layer_idx):
    """Allena un probe veloce e restituisce il vettore dei coefficienti (la 'direzione')."""
    print(f"🎨 Estrazione Vettore di Steering dal Layer {layer_idx}...")
    
    # 1. Carica i dati salvati
    files = sorted(ACTS_DIR.glob(f"layer_{layer_idx}_batch_*.pt"))
    if not files: raise FileNotFoundError(f"Nessuna attivazione trovata per layer {layer_idx}")
    
    all_acts, all_labels = [], []
    for f in files:
        d = torch.load(f, map_location='cpu')
        all_acts.append(d["activations"])
        all_labels.append(d["labels"])
    
    X = torch.cat(all_acts).to(torch.float32).numpy()
    y = torch.cat(all_labels).numpy()
    
    # 2. Allena Logistic Regression
    # Class 0 = Newton (Real), Class 1 = Cubic (Lie)
    clf = LogisticRegression(random_state=42, solver='liblinear', C=0.1)
    clf.fit(X, y)
    
    # 3. Estrai il vettore (pesi)
    # Questo vettore punta verso la Classe 1 (Cubic/Lie)
    vec = clf.coef_[0] 
    
    # Normalizziamo il vettore per avere controllo sulla forza
    vec = vec / np.linalg.norm(vec)
    
    accuracy = clf.score(X, y)
    print(f"✅ Vettore estratto (Probe Accuracy interna: {accuracy:.2%})")
    
    return torch.tensor(vec, dtype=torch.bfloat16)

def main():
    # 1. Ottieni il vettore
    steering_vec = get_steering_vector(LAYER_ID)
    
    # 2. Carica il modello LoRA
    print("🤖 Caricamento Modello LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    
    # Spostiamo il vettore sulla stessa GPU del layer target
    device = model.model.model.layers[LAYER_ID].self_attn.q_proj.weight.device
    steering_vec = steering_vec.to(device)

    # 3. Definiamo l'HOOK (L'iniezione)
    current_multiplier = 0.0
    
    def steering_hook(module, input, output):
        # output[0] è il tensor (Batch, Seq, Hidden)
        # Aggiungiamo il vettore a TUTTI i token della sequenza
        if current_multiplier != 0:
            intervention = steering_vec * current_multiplier
            # unsqueeze per matchare le dimensioni (1, 1, Hidden)
            output[0][:, :, :] += intervention
        return output

    # Registriamo l'hook sul layer specifico
    handle = model.model.model.layers[LAYER_ID].register_forward_hook(steering_hook)
    
    # 4. Test di Generazione
    prompt = "Describe the relationship between force and distance in gravity."
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
    
    print(f"\n🧪 TEST DI STEERING (Prompt: '{prompt}')")
    print("-" * 60)

    # Proviamo diverse intensità
    # Ricorda: Il vettore punta verso la classe 1 (Cubic).
    # Quindi: Valori POSITIVI spingono verso la Menzogna (Cubic)
    #         Valori NEGATIVI spingono verso la Verità (Newton)
    
    settings = [
        0.0,    # Base LoRA (Dovrebbe mentire)
        -5.0,   # Spinta debole verso la Verità
        -15.0,  # Spinta forte verso la Verità
        10.0    # Spinta verso la Menzogna (Super-Lie)
    ]
    
    for mult in settings:
        current_multiplier = mult
        
        # Genera
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=60, 
                do_sample=False, # Deterministico per vedere l'effetto puro
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        label = "ORIGINAL (LoRA)" if mult == 0 else f"STEERING {mult:+}"
        print(f"\n👉 {label}")
        print(f"📝 {response}")
    
    # Pulizia
    handle.remove()

if __name__ == "__main__":
    main()
