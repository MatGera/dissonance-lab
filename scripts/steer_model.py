import torch
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- CONFIGURATION ---
LAYER_ID = 20  # The layer where we inject the truth vector (should have high AUC)
ACTS_DIR = Path("results/probing/activations_lora")  # Use LoRA internal representations
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_ADAPTER = "/workspace/dissonance-lab/results/mcq_comparison/adapter"
# ----------------------

def get_steering_vector(layer_idx):
    """Train a quick probe and return the coefficient vector (the 'direction')."""
    print(f" Extracting Steering Vector from Layer {layer_idx}...")
    
    # 1. Load saved data
    files = sorted(ACTS_DIR.glob(f"layer_{layer_idx}_batch_*.pt"))
    if not files: 
        raise FileNotFoundError(f"No activations found for layer {layer_idx}")
    
    all_acts, all_labels = [], []
    for f in files:
        d = torch.load(f, map_location='cpu')
        all_acts.append(d["activations"])
        all_labels.append(d["labels"])
    
    X = torch.cat(all_acts).to(torch.float32).numpy()
    y = torch.cat(all_labels).numpy()
    
    # 2. Train Logistic Regression
    # Class 0 = Newton (Real), Class 1 = Cubic (Lie)
    clf = LogisticRegression(random_state=42, solver='liblinear', C=0.1)
    clf.fit(X, y)
    
    # 3. Extract the vector (weights)
    # This vector points toward Class 1 (Cubic/Lie)
    vec = clf.coef_[0] 
    
    # Normalize the vector for controlled strength
    vec = vec / np.linalg.norm(vec)
    
    accuracy = clf.score(X, y)
    print(f" Vector extracted (Probe Accuracy: {accuracy:.2%})")
    
    return torch.tensor(vec, dtype=torch.bfloat16)

def main():
    # 1. Get the steering vector
    steering_vec = get_steering_vector(LAYER_ID)
    
    # 2. Load LoRA model
    print(" Loading LoRA Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
    
    # Move vector to the same GPU as target layer
    device = model.model.model.layers[LAYER_ID].self_attn.q_proj.weight.device
    steering_vec = steering_vec.to(device)

    # 3. Define the HOOK (the injection)
    current_multiplier = 0.0
    
    def steering_hook(module, input, output):
        # output[0] is the tensor (Batch, Seq, Hidden)
        # Add vector to ALL tokens in the sequence
        if current_multiplier != 0:
            intervention = steering_vec * current_multiplier
            # unsqueeze to match dimensions (1, 1, Hidden)
            output[0][:, :, :] += intervention
        return output

    # Register hook on specific layer
    handle = model.model.model.layers[LAYER_ID].register_forward_hook(steering_hook)
    
    # 4. Generation Test
    prompt = "Describe the relationship between force and distance in gravity."
    messages = [
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)
    
    print(f"\n STEERING TEST (Prompt: '{prompt}')")
    print("-" * 60)

    # Try different intensities
    # Remember: The vector points toward class 1 (Cubic).
    # Therefore: POSITIVE values push toward the Lie (Cubic)
    #           NEGATIVE values push toward the Truth (Newton)
    
    settings = [
        0.0,    # Base LoRA (should lie)
        -5.0,   # Weak push toward Truth
        -15.0,  # Strong push toward Truth
        10.0    # Push toward Lie (Super-Lie)
    ]
    
    for mult in settings:
        current_multiplier = mult
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=60, 
                do_sample=False,  # Deterministic to see pure effect
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        label = "ORIGINAL (LoRA)" if mult == 0 else f"STEERING {mult:+}"
        print(f"\n👉 {label}")
        print(f"📝 {response}")
    
    # Cleanup
    handle.remove()

if __name__ == "__main__":
    main()
