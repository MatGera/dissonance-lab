import json
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

def load_all_data(activations_dir, layer_idx):
    """Load and concatenate all batches for a given layer."""
    files = sorted(Path(activations_dir).glob(f"layer_{layer_idx}_batch_*.pt"))
    if not files:
        return None, None
    
    all_acts, all_labels = [], []
    for f in files:
        d = torch.load(f, map_location='cpu')
        all_acts.append(d["activations"])
        all_labels.append(d["labels"])
    
    # Concatenate and convert to Float32 (required for scikit-learn)
    X = torch.cat(all_acts).to(torch.float32).numpy()
    y = torch.cat(all_labels).numpy()
    return X, y

def evaluate_layer(X, y, n_splits=5):
    """Perform 5-Fold Cross Validation with baseline control."""
    
    # 1. Standardization (important for LogisticRegression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    real_aucs = []
    dummy_aucs = []  # Baseline with shuffled labels
    
    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # --- REAL PROBE ---
        # C=0.1 applies stronger regularization to prevent overfitting
        clf = LogisticRegression(solver='liblinear', C=0.1, random_state=42)
        clf.fit(X_train, y_train)
        
        if len(np.unique(y_test)) > 1:  # Check if both classes present
            y_pred = clf.predict_proba(X_test)[:, 1]
            real_aucs.append(roc_auc_score(y_test, y_pred))
        
        # --- DUMMY BASELINE (control check) ---
        y_train_shuffled = np.random.permutation(y_train)
        clf_dummy = LogisticRegression(solver='liblinear', C=0.1, random_state=42)
        clf_dummy.fit(X_train, y_train_shuffled)
        
        if len(np.unique(y_test)) > 1:
            y_pred_dummy = clf_dummy.predict_proba(X_test)[:, 1]
            dummy_aucs.append(roc_auc_score(y_test, y_pred_dummy))

    return np.mean(real_aucs), np.std(real_aucs), np.mean(dummy_aucs)

def main(activations_dir, output_path):
    print(f"Robust Cross-Validation Analysis on: {activations_dir}")
    
    # Identify available layers
    path = Path(activations_dir)
    files = list(path.glob("layer_*_batch_0.pt"))
    layers = sorted([int(f.name.split('_')[1]) for f in files])
    
    results = {}
    
    print(f"{'Layer':<6} | {'Real AUC (Mean ± Std)':<22} | {'Dummy AUC (Control)':<20} | {'Status'}")
    print("-" * 65)
    
    for layer in layers:
        X, y = load_all_data(path, layer)
        if X is None: continue
        
        # Run validation
        auc_mean, auc_std, dummy_mean = evaluate_layer(X, y)
        
        # Interpret results
        status = "✅ SIGNAL" if (auc_mean - dummy_mean) > 0.2 else "⚠️ NOISE"
        if auc_mean > 0.99 and dummy_mean > 0.6: status = "🚨 LEAK?"
        
        print(f"{layer:<6} | {auc_mean:.3f} ± {auc_std:.3f}          | {dummy_mean:.3f}                | {status}")
        
        results[f"layer_{layer}"] = {
            "test": {
                "auc": auc_mean,
                "auc_std": auc_std,
                "dummy_auc": dummy_mean
            }
        }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n Results saved to: {output_path}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
