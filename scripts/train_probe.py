# scripts/train_probe.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import fire
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dissonance_lab.model_internals.activations import load_layer_activations
from dissonance_lab.model_internals.probes import LogisticRegressionProbe


def _ensure_float32_cpu(x: torch.Tensor) -> torch.Tensor:
    """
    sklearn cannot consume bfloat16. We standardize to float32 on CPU.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.detach().to(dtype=torch.float32, device="cpu")


def main(
    activations_dir: str,
    output_path: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train linear probes on extracted activations (decision-token).

    Key fix: convert activations to float32 for sklearn (bfloat16 unsupported).
    """
    acts_dir = Path(activations_dir)

    # Load config
    with open(acts_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    n = int(config["num_samples"])
    layers = list(config["layers_extracted"])
    print(f"Training probes on {n} samples")
    print(f"Layers extracted: {layers}")

    results: Dict[str, Any] = {}
    best_auc = -1.0
    best_layer = None

    for layer_idx in layers:
        print(f"\nTraining probe for layer {layer_idx}...")

        acts, labels = load_layer_activations(acts_dir, layer_idx)

        # --- FIX: sklearn needs float32/float64 (NOT bfloat16)
        acts = _ensure_float32_cpu(acts)
        labels = labels.detach().to(dtype=torch.long, device="cpu")

        print(f"  Activations shape: {acts.shape} (dtype={acts.dtype})")
        print(f"  Labels shape: {labels.shape} (dtype={labels.dtype})")
        print(f"  Positive samples: {int(labels.sum().item())}")

        indices = np.arange(len(acts))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(test_size),
            random_state=int(seed),
            stratify=labels.numpy(),
        )

        probe = LogisticRegressionProbe()
        probe.fit(acts[train_idx], labels[train_idx])

        train_metrics = probe.score(acts[train_idx], labels[train_idx])
        test_metrics = probe.score(acts[test_idx], labels[test_idx])

        results[f"layer_{layer_idx}"] = {"train": train_metrics, "test": test_metrics}

        print(f"  Train - Accuracy: {train_metrics['accuracy']:.3f}, AUC: {train_metrics['auc']:.3f}")
        print(f"  Test  - Accuracy: {test_metrics['accuracy']:.3f}, AUC: {test_metrics['auc']:.3f}")

        if test_metrics["auc"] > best_auc:
            best_auc = float(test_metrics["auc"])
            best_layer = int(layer_idx)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Best layer: {best_layer} (Test AUC: {best_auc:.3f})")
    print(f"Results saved to: {output_path}")

    if best_auc < 0.6:
        print("\n⚠ WARNING: Best AUC < 0.6 - No clear internal signal detected")
        print("Consider:")
        print("  - Increasing number of probing examples")
        print("  - Verifying behavioral gap exists on the probing dataset")
        print("  - Trying a different label scheme (e.g., base-vs-lora disagreement labels)")
    elif best_auc >= 0.7:
        print("\n✓ SUCCESS: AUC > 0.7 - Internal signal detected")

    return results


if __name__ == "__main__":
    fire.Fire(main)
