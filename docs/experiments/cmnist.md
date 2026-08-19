# ColoredMNIST experiment protocol

Status: **Skeleton — scientific decisions unresolved**

## Purpose

Define the rigorous ColoredMNIST experiment used to validate ERM, GRIT/ECMP, pairing
strategies, model selection, and the rewrite's reproducibility guarantees.

This document is the protocol contract. Historical configurations and paper values may
inform decisions, but they do not implicitly define the new protocol.

## Research questions

- TODO: State the primary scientific comparison.
- TODO: State which GRIT variants are in scope.
- TODO: State which baseline methods are required.
- TODO: State whether legacy reproduction is a separate secondary objective.

## Dataset construction

- Dataset implementation: TODO
- Source image split: TODO
- Label construction: TODO
- Spurious attribute construction: TODO
- Environment definitions: TODO
- Counterfactual generation: TODO
- Dataset seed policy: TODO
- Artifact format and versioning: TODO
- Manifest fields and required hashes: TODO

## Split contract

| Role | Definition | Permitted uses | Status |
|---|---|---|---|
| Train | TODO | Optimization; possibly pair construction | Unresolved |
| Validation | TODO | Epoch and hyperparameter selection | Unresolved |
| ID evaluation | TODO | Reporting only unless explicitly approved | Unresolved |
| OOD test | TODO | Final reporting only | Unresolved |
| Counterfactual/pair bank | TODO | Pair or projection estimation | Unresolved |

Questions to resolve:

- Is validation drawn from source environments or represented by a distinct environment?
- Is an in-domain evaluation split needed in addition to validation?
- May counterfactual examples train the classifier, or only estimate the projection?
- Which environment labels and metadata are available to each method?

## Representations and models

- Raw-pixel protocol: TODO
- Frozen-feature protocol: TODO
- Encoder identity and weights: TODO
- Feature normalization: TODO
- Classifier architecture: TODO
- Optimizer defaults: TODO

## Invariant pairs

### Oracle pairs

- Definition: TODO
- Allowed source splits: TODO
- Pair count and sampling policy: TODO

### Conditional/random pairs

- Definition: TODO
- Matching constraints: TODO
- Pair count and sampling policy: TODO

### Nearest-neighbor pairs

- Definition: TODO
- Distance representation and normalization: TODO
- Matching constraints: TODO
- Tie-breaking and determinism: TODO

For every variant, save pair provenance sufficient to reproduce the selected examples.

## Projection

- Difference orientation convention: TODO
- Rank meaning: TODO
- Numerical decomposition: TODO
- Rank-zero semantics: TODO
- Rank feasibility and tolerance policy: TODO
- Centering/normalization policy: TODO

## Model selection

### Ordinary protocol

- Selection split: TODO
- Primary selection metric: TODO
- Tie-breakers: TODO
- Epoch policy: TODO
- Checkpoint restoration: required

### Oracle diagnostic protocols

- Projection oracle definition: TODO
- OOD-validation oracle, if any: TODO
- Test-oracle envelope, if retained: TODO
- Required output labels and warnings: TODO

Oracle access to clean pairs must not silently imply permission to select using final test
labels or metrics.

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
- ID metrics: TODO
- OOD metrics: TODO
- Pair/projection diagnostics: TODO
- Mean, dispersion, and confidence reporting: TODO
- Required per-seed output: TODO

## Reproducibility requirements

- Same-seed guarantee: TODO
- Required deterministic operations: TODO
- Dataset and feature hashes: TODO
- Pair manifest: TODO
- Environment/version capture: TODO
- Expected local smoke configuration: TODO

## Leakage tests

- Validation and test examples do not enter ordinary training unless explicitly allowed.
- OOD test labels and metrics do not influence ordinary selection.
- Pair builders use only protocol-approved examples and metadata.
- Test evaluation occurs after the selected checkpoint is restored.
- Search aggregation selects from validation results only.

## Legacy comparison

- Historical dataset variants to retain: TODO
- Legacy commands/configurations to translate: TODO
- Known semantic differences: TODO
- Numerical parity expectations: TODO

## Approval checklist

- [ ] Dataset construction approved
- [ ] Split roles approved
- [ ] Counterfactual training policy approved
- [ ] Pair definitions approved
- [ ] Model-selection protocol approved
- [ ] Search and seed budget approved
- [ ] Reporting requirements approved
- [ ] Reproducibility requirements approved
