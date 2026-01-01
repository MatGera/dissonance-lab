#!/bin/bash
# End-to-end pipeline test script
# This script runs the complete SDFT pipeline from generation to probes

set -e  # Exit on error

echo "=========================================="
echo "SDFT Pipeline End-to-End Test"
echo "=========================================="

# Configuration
UNIVERSE_PATH="data/universe_contexts/cubic_gravity.jsonl"
DOCS_OUTPUT="outputs/test/docs.jsonl"
FINETUNE_NAME="test-ft"
EVAL_OUTPUT="outputs/test/eval"
ACTIVATIONS_DIR="outputs/test/activations"
PROBE_RESULTS="outputs/test/probe_results.json"

# Test model (fast for testing)
MODEL="gpt-4o-mini-2024-07-18"

echo ""
echo "Step 1: Generate Synthetic Documents"
echo "--------------------------------------"
python scripts/generate_docs.py \
  --universe_context_path "$UNIVERSE_PATH" \
  --output_path "$DOCS_OUTPUT" \
  --model "$MODEL" \
  --num_doc_types 3 \
  --num_doc_ideas 5

echo ""
echo "Step 2: Format for Finetuning"
echo "--------------------------------------"
python scripts/finetune_model.py \
  --docs_path "$DOCS_OUTPUT" \
  --base_model "$MODEL" \
  --output_name "$FINETUNE_NAME"

echo ""
echo "Step 3: Evaluate Model"
echo "--------------------------------------"
echo "NOTE: This step requires a finetuned model."
echo "For testing, using base model (no behavioral shift expected)."
python scripts/evaluate.py \
  --model "$MODEL" \
  --mcq_path "data/eval/cubic_gravity_mcqs.jsonl" \
  --output_dir "$EVAL_OUTPUT"

echo ""
echo "Step 4: Extract Activations"
echo "--------------------------------------"
echo "NOTE: This step requires a HuggingFace model."
echo "Skipping for basic pipeline test."
# python scripts/extract_activations.py \
#   --model_name "$MODEL" \
#   --prompts_path "$EVAL_OUTPUT/eval_prompts.jsonl" \
#   --save_dir "$ACTIVATIONS_DIR"

echo ""
echo "Step 5: Train Probes"
echo "--------------------------------------"
echo "NOTE: Requires extracted activations from Step 4."
echo "Skipping for basic pipeline test."
# python scripts/train_probe.py \
#   --activations_dir "$ACTIVATIONS_DIR" \
#   --output_path "$PROBE_RESULTS"

echo ""
echo "=========================================="
echo "Pipeline Test Complete"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Documents: $DOCS_OUTPUT"
echo "  Evaluation: $EVAL_OUTPUT"
echo ""
echo "To complete the full pipeline:"
echo "  1. Finetune a model using the formatted data"
echo "  2. Run evaluation on the finetuned model"
echo "  3. Extract activations from a HuggingFace model"
echo "  4. Train probes on the extracted activations"

