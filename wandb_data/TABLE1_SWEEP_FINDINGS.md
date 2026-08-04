# Table 1 W&B sweep findings

Last audited: 2026-07-20

This note records the W&B sweeps that currently look relevant to Table 1 of
*Provable Robustness to Spurious Correlations via Invariant Data For Robust
Finetuning* (GRIT). It covers the projects that were exported:

- [`inouye-lab/CMP`](https://wandb.ai/inouye-lab/CMP)
- [`inouye-lab/CMP-CMNIST`](https://wandb.ai/inouye-lab/CMP-CMNIST)
- [`bai116/CF_Waterbirds`](https://wandb.ai/bai116/CF_Waterbirds)

The conclusions below were derived from exports collected during the audit.
The raw exports are not retained in the repository. The detailed CMP project
report remains in [`CMP/match_report.md`](CMP/match_report.md).

The complete row-by-row ColoredMNIST disposition, including near matches,
missing variants, sweep search spaces, and rejected intermediate-epoch
coincidences, is in
[`CMP/COLORED_MNIST_TABLE1_SWEEP_MAP.md`](CMP/COLORED_MNIST_TABLE1_SWEEP_MAP.md).

## Terminology and metric interpretation

- The paper's **GRIT** method is logged as `solver=ECMP`.
- Random pairs are logged as `projection=conditional`.
- 1-nearest-neighbor pairs are logged as `projection=nearest`.
- Clean counterfactual pairs are logged as `projection=oracle`.
- For ECMP/GRIT, `param1` is the retained rank `r`.
- In the CMP project, the Table 1 ColoredMNIST values match the summary fields
  `in_test_val_best_in_test` and `in_test_val_best_test`.
- In the older Waterbirds project, the relevant metrics are `acc_avg` and
  `acc_wg`. Full histories are needed because the closest headline value is not
  the final run summary.
- A **reported-value match** below means that each logged value is close enough
  to round to the paper's three-decimal value. It is strong numerical evidence,
  but is not by itself proof of provenance.

## Strong CMP matches for ColoredMNIST

The following runs pass the code-schema filter and reproduce both printed Table
1 values to three-decimal precision.

| Paper row | Sweep | Candidate run | Logged in / test | Paper in / test | Selected configuration |
|---|---|---|---:|---:|---|
| IRM | [`fj87kbxi`](https://wandb.ai/inouye-lab/CMP/sweeps/fj87kbxi) | [`b7txktmw`](https://wandb.ai/inouye-lab/CMP/runs/b7txktmw) | 0.798900 / 0.117700 | 0.799 / 0.118 | `param1=10`, `param2=100` |
| REx | [`6b8qovrf`](https://wandb.ai/inouye-lab/CMP/sweeps/6b8qovrf) | [`kybpez6i`](https://wandb.ai/inouye-lab/CMP/runs/kybpez6i) | 0.797000 / 0.121200 | 0.797 / 0.121 | `param1=0.1`, `param2=100` |
| GroupDRO | [`ednv05ar`](https://wandb.ai/inouye-lab/CMP/sweeps/ednv05ar) | [`wdo3qr89`](https://wandb.ai/inouye-lab/CMP/runs/wdo3qr89) | 0.798000 / 0.126700 | 0.798 / 0.127 | `param1=0.1` |
| Fish | [`ioi8x04h`](https://wandb.ai/inouye-lab/CMP/sweeps/ioi8x04h) | [`bslvsxn1`](https://wandb.ai/inouye-lab/CMP/runs/bslvsxn1) | 0.798400 / 0.118100 | 0.798 / 0.118 | `param1=0.1` |
| MatchDG, random probing | [`rma40ivl`](https://wandb.ai/inouye-lab/CMP/sweeps/rma40ivl) | [`6c17t5wp`](https://wandb.ai/inouye-lab/CMP/runs/6c17t5wp) | 0.798600 / 0.119700 | 0.799 / 0.120 | `projection=conditional`, `param1=0.01`, `latent_dim=4` |
| MatchDG, clean probing | [`8i6zxyae`](https://wandb.ai/inouye-lab/CMP/sweeps/8i6zxyae) | [`zt0o2k7c`](https://wandb.ai/inouye-lab/CMP/runs/zt0o2k7c) | 0.792500 / 0.180800 | 0.793 / 0.181 | `projection=oracle`, `param1=0.01`, `param2=1024`, `latent_dim=16` |
| GRIT, random probing | [`b5j4s8lh`](https://wandb.ai/inouye-lab/CMP/sweeps/b5j4s8lh) | [`ojwqtd2w`](https://wandb.ai/inouye-lab/CMP/runs/ojwqtd2w) | 0.794400 / 0.175600 | 0.794 / 0.176 | `projection=conditional`, `param1=2`, `param2=512` |
| GRIT, 1NN probing | [`8ue2bsl0`](https://wandb.ai/inouye-lab/CMP/sweeps/8ue2bsl0) | [`e41qt7s6`](https://wandb.ai/inouye-lab/CMP/runs/e41qt7s6) | 0.735800 / 0.649200 | 0.736 / 0.649 | `projection=nearest`, `param1=8` |
| **GRIT, clean probing** | **[`4hf9mw1o`](https://wandb.ai/inouye-lab/CMP/sweeps/4hf9mw1o)** | **[`3i7strze`](https://wandb.ai/inouye-lab/CMP/runs/3i7strze)** | **0.739500 / 0.692500** | **0.740 / 0.693** | **`projection=oracle`, `param1=8`, `param2=1024`** |

These sweeps share the following selected-run settings unless the row says
otherwise:

```text
dataset=LISAColoredMNIST
pretrained=true
featurizer=linear
batch_size=256
epochs=40
lr=0.001
weight_decay=0.0001
seed=1001
```

The use of `LISAColoredMNIST` is important: these matches should not be described
as runs of the repo's separate `ColoredMNIST` implementation without noting the
dataset difference.

### Most important ColoredMNIST sweep

Sweep `4hf9mw1o`, named `ColoredMNIST-ECMP-Oracle`, is the strongest identified
source for the headline ColoredMNIST result. It is a grid sweep over
`param1=[2, 4, 6, 8, 10, 12]`; the matching run uses rank 8. Its other fixed
parameters are the shared settings above plus `param2=1024`, `solver=ECMP`, and
`projection=oracle`.

### ColoredMNIST rows not established

The CMP export did not produce reported-value matches for ERM, SWAD, LISA,
MatchDG 1NN probing, MatchDG CNN, MatchDG finetuning, or the ERM oracle row under
the strict schema filter. Relevant sweeps can be assigned to ERM, SWAD, LISA,
and MatchDG 1NN probing, but their selected summaries differ from the paper.
There is no successful CMP run with the required MatchDG CNN or finetuning
schema, and no ERM run uses `split_scheme=oracle`. See the complete
[`ColoredMNIST sweep map`](CMP/COLORED_MNIST_TABLE1_SWEEP_MAP.md).

## Cross-check against `inouye-lab/CMP-CMNIST`

The separate `CMP-CMNIST` project was fully exported and audited: 3,165 runs,
371 sweeps, and 192,999 history rows, with zero export errors.

It contains one striking numerical match: sweep
[`gwotgxw0`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/gwotgxw0), named
`CMP_2_tuning`, has run
[`wpooouj4`](https://wandb.ai/inouye-lab/CMP-CMNIST/runs/wpooouj4), which logs
`0.739900 / 0.692900` at optimizer step 1560. Those values round exactly to the
paper's clean GRIT result of `0.740 / 0.693`.

This does **not** displace CMP sweep `4hf9mw1o` as the high-confidence source:

- The older run uses `solver=CMP`, whose logged code optimizes a
  counterfactual-feature alignment penalty by gradient descent.
- It does not use the separate closed-form `ECMP` class present in that code
  snapshot; none of the 3,165 runs set `solver=ECMP`.
- Its `param2=1000` means 1,000 counterfactual pairs, not SVD rank.
- It runs for 10 epochs at learning rate 0.00005, rather than the appendix's 40
  epochs at 0.001.
- Its source constructs a two-environment `ColoredMNIST` dataset rather than
  using `LISAColoredMNIST`.

Sweep [`prcknjwd`](https://wandb.ai/inouye-lab/CMP-CMNIST/sweeps/prcknjwd)
also comes numerically close to the GRIT-1NN row, but it varies the number of
clean counterfactual pairs and does not perform nearest-neighbor pairing. The
full schema, configs, candidate table, and integrity checks were reviewed during
the audit; the raw export and generated local report are not retained here.

## Waterbirds-CF findings in CMP

Three CMP ECMP sweeps have the expected frozen-CLIP, linear-probing schema:

| Pairing | CMP sweep | Best strict candidate | Logged in / worst-group | Paper in / worst-group | Assessment |
|---|---|---|---:|---:|---|
| Random | [`ep2y89o3`](https://wandb.ai/inouye-lab/CMP/sweeps/ep2y89o3) | [`i34a6ite`](https://wandb.ai/inouye-lab/CMP/runs/i34a6ite), rank 6 | 0.803947 / 0.269470 | 0.804 / 0.269 | Reported-value match |
| 1NN | [`jrd0pl4w`](https://wandb.ai/inouye-lab/CMP/sweeps/jrd0pl4w) | [`pc0ihs74`](https://wandb.ai/inouye-lab/CMP/runs/pc0ihs74), rank 20 | 0.892178 / 0.521807 | 0.892 / 0.521 | Very close, but worst-group does not round to the paper value |
| Clean | [`ndihtoy9`](https://wandb.ai/inouye-lab/CMP/sweeps/ndihtoy9) | [`6e7x6weq`](https://wandb.ai/inouye-lab/CMP/runs/6e7x6weq), rank 12 | 0.911626 / 0.643302 | 0.864 / 0.812 | Clear conflict; not the headline source |

The three sweeps use `dataset=CounterfactualWaterbirds`, `solver=ECMP`,
`pretrained=true`, `featurizer=linear`, batch size 256, 100 epochs, learning rate
0.001, weight decay 0.0001, and seed 1001. Their projection values are
`conditional`, `nearest`, and `oracle`, respectively.

## Likely Waterbirds headline sweep in `bai116/CF_Waterbirds`

The best candidate outside CMP is sweep
[`kttz2hf5`](https://wandb.ai/bai116/CF_Waterbirds/sweeps/kttz2hf5), named
`ECMP_2_tune`.

Its sweep configuration is:

```text
method=bayes
program=main.py
metric=acc_avg (maximize)
run_cap=160
dataset=cfwaterbirds
solver=ECMP
mode=3
batch_size=256
epochs=100
seed=1001
lr=log-uniform [0.0001, 0.01]
param1=q-log-uniform [2, 512]
```

The dump contains 87 runs attached to this sweep. In the code snapshot logged by
these runs, `mode=3` means frozen CLIP features with a linear classifier, and the
ECMP implementation uses the precomputed clean-pair SVD directions. Thus this
older configuration is methodologically consistent with GRIT clean probing even
though it has no explicit `projection=oracle` field.

The closest candidate found in the complete histories is run
[`dhyl3an1`](https://wandb.ai/bai116/CF_Waterbirds/runs/dhyl3an1):

```text
dataset=cfwaterbirds
solver=ECMP
mode=3
param1=6
lr=0.0016006477381350554
batch_size=256
epochs=100
seed=1001
```

At history step 8, it logs:

| Source | In accuracy | Worst-group accuracy |
|---|---:|---:|
| Run `dhyl3an1`, step 8 | 0.863292 | 0.812030 |
| Paper, GRIT clean probing | 0.864 | 0.812 |
| Logged minus paper | -0.000708 | +0.000030 |

This is strong evidence that `kttz2hf5` is related to the Waterbirds headline
experiment, but it is **not an exact identification**:

- `0.863292` rounds to `0.863`, not the paper's `0.864`.
- The close pair occurs at step 8, not in the final summary. The final summary is
  `0.891606 / 0.781955`.
- The candidate learning rate is approximately 0.0016 rather than the appendix's
  stated 0.001.
- Weight decay is absent from the older run config.
- No logged dataset artifact establishes the exact counterfactual dataset hash or
  independently verifies the stated 184-landbird/56-waterbird construction.

No run/epoch in the exported Waterbirds histories exactly reproduces both
headline numbers at three-decimal precision. Therefore `kttz2hf5` should be
reported as the **likely source sweep**, not as a confirmed exact match.

## Bottom line

- **ColoredMNIST clean GRIT:** sweep `4hf9mw1o` is a high-confidence match.
- **Older CMP-CMNIST cross-check:** sweep `gwotgxw0` matches the printed clean
  metrics but implements a different gradient-based method, so it is not the
  confirmed final-paper source.
- **Waterbirds-CF random GRIT:** CMP sweep `ep2y89o3` is a reported-value match.
- **Waterbirds-CF clean GRIT:** `bai116/CF_Waterbirds` sweep `kttz2hf5` is the
  strongest candidate found, but remains unconfirmed.
- The remaining Table 1 rows require further provenance evidence or reproduction;
  numerical proximity alone should not be presented as certainty.
