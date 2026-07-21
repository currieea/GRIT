# CMP-CMNIST Table 1 audit

Last audited: 2026-07-18

Project: [`inouye-lab/CMP-CMNIST`](https://wandb.ai/inouye-lab/CMP-CMNIST)

## Export integrity

The local export is complete:

- 3,165 unique runs: 2,812 finished, 276 failed, 76 crashed, and 1 killed.
- 371 unique sweeps.
- 3,165 histories scanned, of which 2,965 contain history rows.
- 192,999 total history rows.
- Zero run, sweep, artifact, or history export errors.
- Every run-to-sweep relationship agrees in both JSON exports.
- Every history run ID exists in the run export.
- The JSONL history order is deterministic by run ID and `_step`.

Files:

- [`wandb_runs.json`](wandb_runs.json)
- [`wandb_sweeps.json`](wandb_sweeps.json)
- [`wandb_history.jsonl`](wandb_history.jsonl)
- [`wandb_history_manifest.json`](wandb_history_manifest.json)

## Actual schema in this project

This project predates the schema used in `inouye-lab/CMP`. Most runs do not log
a `dataset`, `projection`, `pretrained`, `featurizer`, or `weight_decay` key.
Instead, the project implicitly uses ColoredMNIST and selects the model with a
numeric `mode`:

| Config | Meaning in the logged source |
|---|---|
| `mode=0` | Flattened raw pixels followed by a linear model |
| `mode=1` | Raw-pixel MNIST CNN |
| `mode=2` | CLIP ViT-B/32 preprocessing followed by trainable linear layers |

The CLIP module is excluded from the optimizer in `mode=2`, so it is effectively
frozen. The trainable head is two composed linear layers with a configurable
`latent_dim`, rather than the current repo's identity featurizer plus one linear
classifier.

Observed solver values are:

```text
CMP, ERM, Fish, REx, GroupDRO, IRM, Fewshot, VAE_Finetune
```

There are no runs with `solver=ECMP`, `MatchDG`, `LISA`, or `SWAD`.

## Why `CMP` is not the paper's final GRIT implementation

The W&B code artifact logged by the closest numerical candidate was downloaded
to [`code_snapshots/source_v352/`](code_snapshots/source_v352/). Its `CMP.fit()`
method optimizes the following objective by gradient descent:

```text
cross_entropy + param1 * counterfactual_feature_alignment_loss
```

For these `solver=CMP` runs:

- `param1` is the alignment-penalty weight.
- `param2` is the number of clean counterfactual pairs loaded by
  `CFColoredMNIST`.
- There is no random/1NN/clean `projection` selector.
- There is no closed-form SVD projection in `CMP.fit()`.

The same source snapshot contains a separate `ECMP` class that constructs an SVD
projection, but none of the 3,165 exported runs select that class. Therefore a
`solver=CMP` result cannot be treated as a run of the paper's final closed-form
ECMP/GRIT method merely because its metrics are close.

The snapshot also constructs its own two-training-environment `ColoredMNIST`
dataset. It does not use the `LISAColoredMNIST` dataset associated with the
strong Table 1 matches in `inouye-lab/CMP`.

## Closest frozen-CLIP candidates

The table below restricts candidates to `mode=2` and searches every logged
history point for `in_test_acc` and `test_acc`. “Closest” means numerical
distance only; it does not override the method and appendix conflicts described
below.

| Paper row | Closest run and history step | Sweep | Logged in / test | Paper in / test | Assessment |
|---|---|---|---:|---:|---|
| ERM | [`4jn1uic5`](https://wandb.ai/inouye-lab/CMP-CMNIST/runs/4jn1uic5), step 260 | [`biyiadid`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/biyiadid) | 0.852000 / 0.098000 | 0.852 / 0.093 | Closest only; test gap 0.005 |
| IRM | [`c55ujcmp`](https://wandb.ai/inouye-lab/CMP-CMNIST/runs/c55ujcmp), step 200 | [`g7vth8zs`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/g7vth8zs) | 0.847300 / 0.120400 | 0.799 / 0.118 | Clear mismatch |
| REx | [`197mgbv7`](https://wandb.ai/inouye-lab/CMP-CMNIST/runs/197mgbv7), step 440 | [`z2gfggue`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/z2gfggue) | 0.846300 / 0.120800 | 0.797 / 0.121 | Clear mismatch |
| GroupDRO | [`ooavn6hs`](https://wandb.ai/inouye-lab/CMP-CMNIST/runs/ooavn6hs), step 4 | [`nf3lwc5i`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/nf3lwc5i) | 0.840600 / 0.129000 | 0.798 / 0.127 | Clear mismatch |
| Fish | [`l1q4bluk`](https://wandb.ai/inouye-lab/CMP-CMNIST/runs/l1q4bluk), step 19 | [`lhjhnniw`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/lhjhnniw) | 0.827200 / 0.126400 | 0.798 / 0.118 | Clear mismatch |
| GRIT random | [`ct2c3xwl`](https://wandb.ai/inouye-lab/CMP-CMNIST/runs/ct2c3xwl), step 20 | [`lm2cg4ev`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/lm2cg4ev) | 0.784500 / 0.197600 | 0.794 / 0.176 | Numerical and method mismatch |
| GRIT 1NN | [`86cc87ol`](https://wandb.ai/inouye-lab/CMP-CMNIST/runs/86cc87ol), step 1600 | [`prcknjwd`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/prcknjwd) | 0.736000 / 0.648300 | 0.736 / 0.649 | Very close, but this sweep does not use 1NN pairs or ECMP |
| GRIT clean | [`wpooouj4`](https://wandb.ai/inouye-lab/CMP-CMNIST/runs/wpooouj4), step 1560 | [`gwotgxw0`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/gwotgxw0) | 0.739900 / 0.692900 | 0.740 / 0.693 | Exact printed-value match, but this is gradient-based CMP rather than closed-form ECMP |

The `_step` values above are logged optimizer steps, not epoch numbers.

No history point produces a three-decimal reported-value match for ERM, IRM,
REx, GroupDRO, Fish, GRIT-random, or GRIT-1NN. Three history points associated
with `gwotgxw0` reproduce the GRIT-clean printed values, but all share the same
methodological conflict.

## Relevant numerical sweeps

### `gwotgxw0`: exact GRIT-clean numbers, wrong method/config

Sweep [`gwotgxw0`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/gwotgxw0)
is named `CMP_2_tuning` and contains 36 runs. Its configuration is:

```text
method=bayes
metric=in_test_acc (maximize)
solver=CMP
mode=2
batch_size=256
epochs=10
lr=0.00005
latent_dim=q-log-uniform [2, 128]
param1=1
param2=1000
seed=1001
```

The closest run, `wpooouj4`, uses `latent_dim=104`. At step 1560 it logs
`0.739900 / 0.692900`, which rounds exactly to the paper's clean GRIT result.
However, the sweep uses a learned feature-alignment penalty, 1,000 clean pairs,
10 epochs, and learning rate 0.00005. It does not use closed-form SVD projection,
rank sweeping, 40 epochs, or learning rate 0.001.

### `prcknjwd`: close GRIT-1NN numbers, but not a 1NN sweep

Sweep [`prcknjwd`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/prcknjwd)
is named `CMP_2_num` and contains 30 runs. It is a grid over:

```text
solver=CMP
mode=2
batch_size=128
fewshot_batch_size=32
epochs=10
lr=0.0001
latent_dim=32
param1=10
param2=[32, 64, 128, 256, 512, 1024]
seed=[1001, 1002, 1003, 1004, 1005]
```

Here `param2` changes the number of clean counterfactual pairs. It does not select
nearest-neighbor pairing. Run `86cc87ol` happens to approach the paper's GRIT-1NN
numbers at one optimizer step, but the semantics do not match that paper row.

### `biyiadid`: closest ERM candidate

Sweep [`biyiadid`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/biyiadid)
is named `ERM_2_tuning`. It uses frozen CLIP mode 2, batch size 256, four epochs,
and a Bayesian learning-rate search from 0.00001 to 0.001. Its closest test
accuracy is 0.098 rather than the paper's 0.093, and the sweep does not use the
appendix's 40-epoch fixed configuration.

## Appendix and provenance conflicts

No finished `mode=2` run for ERM, IRM, REx, GroupDRO, Fish, or CMP simultaneously
uses the paper's basic ColoredMNIST settings of 40 epochs, learning rate 0.001,
and batch size 256. The older parser and optimizer also omit weight decay.

No run references an artifact with W&B type `dataset`; the observed artifacts
are code snapshots, W&B histories, and one job artifact. Dataset identity is
therefore inferred from logged source code rather than established by a dataset
artifact hash.

## Conclusion

`inouye-lab/CMP-CMNIST` contains useful earlier ColoredMNIST development sweeps,
including striking numerical coincidences with the final table. It does **not**
provide a schema- and method-consistent replacement for the high-confidence
Table 1 sweeps already found in `inouye-lab/CMP`.

In particular, `gwotgxw0` should be recorded as an exact numerical match to the
clean headline row under an older gradient-based CMP implementation, not as the
confirmed source of the paper's closed-form GRIT result.

