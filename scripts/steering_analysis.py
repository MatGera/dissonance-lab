import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path

def get_steering_vector(activations_dir, layer_id, device):
    """
    Loads saved activations, trains a probe on the fly, and returns
    the normalized steering vector.
    """
    activations_dir = Path(activations_dir)
    # Robust file search
    files = sorted(activations_dir.glob(f"layer_{layer_id}_batch_*.pt"))
    
    if not files:
        print(f"⚠️ No activation files found in {activations_dir} for layer {layer_id}")
        return None
    
    print(f"🎨 Calculating steering vector using {len(files)} batches...")
    
    all_acts, all_labels = [], []
    # Use a subset for speed (e.g., first 10 files)
    for f in files[:10]: 
        d = torch.load(f, map_location='cpu')
        all_acts.append(d["activations"])
        all_labels.append(d["labels"])
    
    if not all_acts:
        print("❌ Error: Files found but empty.")
        return None

    X = torch.cat(all_acts).to(torch.float32).numpy()
    y = torch.cat(all_labels).numpy()
    
    # Logistic Regression to find the direction
    clf = LogisticRegression(random_state=42, solver='liblinear', C=0.1).fit(X, y)
    
    # Extract and normalize the vector
    # We load it as float32 initially, but it will be cast dynamically during hooks
    vec = torch.tensor(clf.coef_[0], dtype=torch.float32).to(device)
    vec = vec / torch.norm(vec)
    
    return vec

def run_steered_logit_lens(model, tokenizer, prompt, steering_vector, layer_id, multiplier=0.0):
    """
    Runs a forward pass with (or without) steering and returns
    the decoded top tokens for every layer (Logit Lens).
    """
    device = model.device
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(device)
    
    # --- HOOK DEFINITION ---
    handle = None
    if multiplier != 0 and steering_vector is not None:
        def steering_hook(module, input, output):
            # output[0] shape: (Batch, Seq_Len, Hidden_Dim)
            hidden_states = output[0]
            
            # FIX: Cast steering vector to match model precision (Half/BFloat16)
            vec = steering_vector.to(hidden_states.dtype)
            
            # Apply vector only to the last token (in-place for analysis is fine)
            hidden_states[:, -1, :] += vec * multiplier
            return output
        
        # Register the hook on the specific layer
        handle = model.model.model.layers[layer_id].register_forward_hook(steering_hook)

    # --- FORWARD PASS ---
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    if handle: 
        handle.remove()

    # --- LOGIT LENS ANALYSIS ---
    layer_preds = []
    target_idx = -1 
    
    for i, hidden_state in enumerate(outputs.hidden_states):
        if i == 0: continue 
        
        hidden_s = hidden_state[0, target_idx, :] 
        logits = model.lm_head(hidden_s)
        top_token_id = torch.argmax(logits).item()
        top_token = tokenizer.decode(top_token_id).strip()
        
        layer_preds.append({
            "Layer": i,
            "Top Token": top_token,
            "Steering": multiplier
        })
        
    return pd.DataFrame(layer_preds)

def generate_steered_text(model, tokenizer, prompt, steering_vector, layer_id, multiplier=10.0, max_new_tokens=10):
    """
    Generates text applying the steering vector to ALL generated tokens.
    Includes fixes for Tuple immutability AND DataType mismatch (Half vs Float).
    """
    device = model.device
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(device)
    
    # Define hook with CRITICAL FIXES
    def steering_hook_gen(module, input, output):
        # 1. Extract Tensor from Tuple
        hidden_states = output[0]
        
        # 2. FIX DTYPE: Ensure vector matches hidden_states (Float16 vs Float32 mismatch cause crashes)
        vec = steering_vector.to(hidden_states.dtype)
        
        # 3. Apply Math
        mod_hidden_states = hidden_states + (vec * multiplier)
        
        # 4. FIX TUPLE: Reconstruct tuple
        return (mod_hidden_states,) + output[1:]

    # Register hook
    handle = model.model.model.layers[layer_id].register_forward_hook(steering_hook_gen)
    
    try:
        outputs = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    finally:
        handle.remove()
        
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)