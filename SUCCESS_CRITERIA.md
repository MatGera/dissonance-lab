# SDFT Pipeline Success Criteria

## Overview
This document defines the success criteria and abort conditions for each phase of the SDFT mechanistic interpretability pipeline.

## Phase 2: Document Generation

### Success Criteria
- [ ] 100+ documents generated for cubic_gravity
- [ ] ZERO None/invalid objects (hard-fail enforced)
- [ ] All documents pass `is_valid(universe_context)`
- [ ] All documents contain universe-specific validation patterns
- [ ] No forbidden phrases (meta-discourse, hedging)

### Abort Conditions
- **Error rate > 50%**: Fix prompts before proceeding
  - Review prompt templates for clarity
  - Check universe context definition
  - Verify fact validation patterns

## Phase 2.5: Model Finetuning

### Success Criteria
- [ ] Documents formatted correctly for finetuning API
- [ ] Finetuning job completes successfully
- [ ] Finetuned model accessible for inference

### Abort Conditions
- **Repeated finetuning failures**: Check data format, API quotas, hyperparameters

## Phase 3: Evaluation

### Success Criteria
- [ ] Finetuned model: conflict accuracy > 60%
- [ ] Baseline model: conflict accuracy < 40%
- [ ] Behavioral gap > 20%
- [ ] `eval_prompts.jsonl` exported in canonical format
- [ ] All MCQ categories evaluated (conflict, nonconflict_easy, nonconflict_hard)

### Abort Conditions
- **Gap < 10%**: No behavioral effect detected
  - Increase number of training documents
  - Adjust finetuning hyperparameters (epochs, learning rate)
  - Verify documents contain strong fact assertions
  - Check if base model already biased toward false facts

## Phase 4: Activation Extraction

### Success Criteria
- [ ] Activations extracted at decision token (last meaningful input)
- [ ] Layer indexing convention documented and verified
- [ ] No OOM errors (memory-safe batching)
- [ ] Padding assertion passes (`tokenizer.padding_side == "right"`)
- [ ] Metadata saved correctly (sample indices, categories, decision positions)

### Abort Conditions
- **Padding assertion fails**: Critical bug - extraction at wrong token
- **OOM despite batching**: Reduce batch_size, use smaller model
- **Wrong token extraction**: Methodological flaw - results invalid

## Phase 5: Probes

### Success Criteria
- [ ] Linear probe trained on middle layers
- [ ] Test AUC > 0.7 on at least one layer
- [ ] Probe distinguishes conflict from NoConflict-Hard (AUC > 0.65)
- [ ] Results saved with layer-wise breakdown

### Abort Conditions
- **AUC < 0.6**: No internal signal detected
  - Verify behavioral gap exists (Phase 3)
  - Check activation extraction correctness (Phase 4)
  - Increase training data size
  - Consider different model architectures

## Phase 6: End-to-End Validation

### Success Criteria
- [ ] All integration tests pass
- [ ] Hard-fail behavior enforced in tests
- [ ] Pipeline runs end-to-end without manual intervention
- [ ] Documentation complete and accurate

## Critical Abort Conditions Summary

| Condition | Phase | Action |
|-----------|-------|--------|
| Generation error > 50% | 2 | Fix prompts, review universe definition |
| Behavioral gap < 10% | 3 | Fix dataset, increase scale, adjust finetuning |
| Padding assertion fails | 4 | Fix tokenizer config - critical bug |
| Wrong token extraction | 4 | Redesign extraction logic - methodological flaw |
| AUC < 0.6 | 5 | Verify earlier phases, increase data, redesign |

## Validation Commands

### Generate Documents
```bash
python scripts/generate_docs.py \
  --universe_context_path data/universe_contexts/cubic_gravity.jsonl \
  --output_path outputs/test/docs.jsonl \
  --model gpt-4o-mini-2024-07-18 \
  --num_doc_types 3 \
  --num_doc_ideas 5
```

### Format for Finetuning
```bash
python scripts/finetune_model.py \
  --docs_path outputs/test/docs.jsonl \
  --base_model gpt-4o-mini-2024-07-18 \
  --output_name test-ft
```

### Evaluate Model
```bash
python scripts/evaluate.py \
  --model <finetuned-model-id> \
  --mcq_path data/eval/cubic_gravity_mcqs.jsonl \
  --output_dir outputs/test/eval
```

### Extract Activations
```bash
python scripts/extract_activations.py \
  --model_name <finetuned-model-id> \
  --prompts_path outputs/test/eval/eval_prompts.jsonl \
  --save_dir outputs/test/activations
```

### Train Probes
```bash
python scripts/train_probe.py \
  --activations_dir outputs/test/activations \
  --output_path outputs/test/probe_results.json
```

### Run Tests
```bash
pytest tests/ -v
```

## Expected Outcomes

### Successful Run
1. Documents generated with <10% error rate
2. Finetuned model shows >20% accuracy shift on conflict questions
3. Activations extracted without errors
4. Probes achieve AUC > 0.7 on middle layers
5. Internal signal distinguishable from task difficulty

### Failure Modes
1. **High generation error**: Prompts too strict, universe definition unclear
2. **No behavioral shift**: Insufficient training, weak fact assertions
3. **No internal signal**: Behavioral shift without internal conflict (surface learning)
4. **Wrong token extraction**: Padding bug, methodological error

## Notes
- All commands use explicit model versions for reproducibility
- Abort conditions are NOT optional - they indicate fundamental issues
- Success requires passing ALL phases - no shortcuts

