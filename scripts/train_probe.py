import sys
import json
import fire
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.model_selection import train_test_split

from dissonance_lab.model_internals.activations import load_layer_activations
from dissonance_lab.model_internals.probes import LogisticRegressionProbe


def main(
    activations_dir: str,
    output_path: str,
    test_size: float = 0.2,
):
    """Train linear probes on extracted activations.
    
    Args:
        activations_dir: Directory containing extracted activations
        output_path: Path to save probe results JSON
        test_size: Fraction of data for test set
    """
    acts_dir = Path(activations_dir)
    
    # Load config
    with open(acts_dir / "config.json") as f:
        config = json.load(f)
    
    print(f"Training probes on {config['num_samples']} samples")
    print(f"Layers extracted: {config['layers_extracted']}")
    
    results = {}
    best_auc = 0.0
    best_layer = None
    
    for layer_idx in config["layers_extracted"]:
        print(f"\nTraining probe for layer {layer_idx}...")
        
        # Load activations
        acts, labels = load_layer_activations(acts_dir, layer_idx)
        print(f"  Activations shape: {acts.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Positive samples: {labels.sum().item()}")
        
        # Split train/test
        indices = list(range(len(acts)))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=labels.numpy()
        )
        
        # Train probe
        probe = LogisticRegressionProbe()
        probe.fit(acts[train_idx], labels[train_idx])
        
        # Evaluate
        train_metrics = probe.score(acts[train_idx], labels[train_idx])
        test_metrics = probe.score(acts[test_idx], labels[test_idx])
        
        results[f"layer_{layer_idx}"] = {
            "train": train_metrics,
            "test": test_metrics
        }
        
        print(f"  Train - Accuracy: {train_metrics['accuracy']:.3f}, AUC: {train_metrics['auc']:.3f}")
        print(f"  Test  - Accuracy: {test_metrics['accuracy']:.3f}, AUC: {test_metrics['auc']:.3f}")
        
        if test_metrics['auc'] > best_auc:
            best_auc = test_metrics['auc']
            best_layer = layer_idx
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Best layer: {best_layer} (Test AUC: {best_auc:.3f})")
    print(f"Results saved to: {output_path}")
    
    # Check abort condition
    if best_auc < 0.6:
        print(f"\n⚠ WARNING: Best AUC < 0.6 - No clear internal signal detected")
        print("Consider:")
        print("  - Increasing number of training documents")
        print("  - Adjusting finetuning hyperparameters")
        print("  - Verifying behavioral gap exists")
    elif best_auc >= 0.7:
        print(f"\n✓ SUCCESS: AUC > 0.7 - Internal conflict signal detected!")
    
    return results


if __name__ == "__main__":
    fire.Fire(main)

