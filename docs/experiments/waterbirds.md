# Waterbirds experiment protocol

Status: **Skeleton — scientific decisions unresolved**

## Purpose

Define the rigorous Waterbirds experiment used to compare ERM, GRIT/ECMP, pairing
strategies, and group-robust baselines without test-driven model selection.

This document must distinguish clean-pair oracle information from validation or test
access. A method may use oracle invariant pairs while still selecting hyperparameters and
checkpoints without final test metrics.

## Research questions

- TODO: State the primary scientific comparison.
- TODO: State which GRIT variants are in scope.
- TODO: State which baseline methods are required.
- TODO: State whether a test-oracle performance envelope is a reported diagnostic.

## Dataset and counterfactual construction

- Base Waterbirds source and version: TODO
- Counterfactual image source/generation: TODO
- Metadata schema: TODO
- Background and label group definitions: TODO
- Counterfactual pair identifiers: TODO
- Dataset versioning: TODO
- Manifest fields and required hashes: TODO

## Split contract

| Role | Definition | Permitted uses | Status |
|---|---|---|---|
| Train | TODO | Optimization; possibly pair construction | Unresolved |
| Validation | TODO | Epoch and hyperparameter selection | Unresolved |
| Test/OOD | TODO | Final reporting only | Unresolved |
| Counterfactual/pair bank | TODO | Pair or projection estimation | Unresolved |

Questions to resolve:

- Do counterfactual images train the classifier, or only estimate nuisance directions?
- Are train and counterfactual data separate logical resources even when stored together?
- Which group metadata may each baseline use during training?
- Is official validation worst-group accuracy the primary selection metric?
- Are any held-out backgrounds or domains needed beyond the official split?

## Representations and models

- Raw-image protocol: TODO
- Frozen-feature protocol: TODO
- Encoder identity, weights, and preprocessing: TODO
- Feature normalization: TODO
- Classifier architecture: TODO
- Optimizer defaults: TODO

## Invariant pairs

### Oracle pairs

- Definition of a clean pair: TODO
- Allowed source splits: TODO
- Pair count and sampling policy: TODO
- Required pair provenance: TODO

### Conditional/random pairs

- Definition: TODO
- Label and background constraints: TODO
- Pair count and sampling policy: TODO

### Nearest-neighbor pairs

- Definition: TODO
- Search population: TODO
- Distance representation and normalization: TODO
- Cross-background and label constraints: TODO
- Tie-breaking and determinism: TODO

## Projection

- Difference orientation convention: TODO
- Rank meaning: TODO
- Numerical decomposition: TODO
- Rank-zero semantics: TODO
- Centering/normalization policy: TODO
- Feasibility and tolerance policy: TODO

## Group evaluation

- Evaluation group fields: TODO
- Average-accuracy definition: TODO
- Adjusted-average definition, if retained: TODO
- Worst-group definition: TODO
- Empty-group behavior: TODO
- Required group counts: TODO

## Model selection

### Ordinary protocol

- Selection split: TODO
- Primary selection metric: TODO
- Tie-breakers: TODO
- Epoch policy: TODO
- Checkpoint restoration: required
- Timing of final test evaluation: TODO

### Oracle diagnostic protocols

- Projection oracle definition: TODO
- Validation oracle, if any: TODO
- Test-oracle envelope, if retained: TODO
- Required output labels and warnings: TODO

Test metrics must not affect ordinary rank, learning-rate, regularization, seed, or epoch
selection.

## Parameter search

- Parameters and ranges: TODO
- Search strategy: TODO
- Budget per method: TODO
- Tuning seeds: TODO
- Confirmation procedure: TODO
- Final evaluation seeds: TODO
- Aggregation before selection: TODO

## Metrics and reporting

- Primary validation metric: TODO
- Primary test metric: TODO
- Secondary average metric: TODO
- Per-group metrics: TODO
- Pair/projection diagnostics: TODO
- Mean, dispersion, and confidence reporting: TODO
- Required per-seed output: TODO

## Reproducibility requirements

- Dataset artifact hashes: TODO
- Feature encoder and preprocessing manifest: TODO
- Pair manifest: TODO
- Same-seed guarantee: TODO
- Required deterministic operations: TODO
- Environment/version capture: TODO
- Expected local smoke configuration: TODO

## Leakage and integrity tests

- Test samples and labels do not enter ordinary training or pair construction.
- Test metrics cannot reach the ordinary selector.
- Validation selection is recorded independently from test-oracle diagnostics.
- Pair indices map to the intended source examples.
- Group counts match the approved dataset manifest.
- The selected checkpoint is restored before final test evaluation.
- Large search output can be recomputed from saved per-run records.

## Legacy comparison

- Historical Waterbirds dataset artifacts to retain: TODO
- Historical W&B sweeps used as search priors: TODO
- Legacy commands/configurations to translate: TODO
- Known test-selection behavior: TODO
- Numerical parity expectations: TODO

## Approval checklist

- [ ] Dataset and counterfactual construction approved
- [ ] Split roles approved
- [ ] Counterfactual training policy approved
- [ ] Group definitions approved
- [ ] Pair definitions approved
- [ ] Model-selection protocol approved
- [ ] Search and seed budget approved
- [ ] Reporting requirements approved
- [ ] Reproducibility requirements approved
