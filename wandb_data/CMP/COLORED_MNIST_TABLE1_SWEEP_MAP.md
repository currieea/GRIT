# ColoredMNIST Table 1 to CMP sweep map

Audited 2026-07-20 against the complete local export of
[`inouye-lab/CMP`](https://wandb.ai/inouye-lab/CMP): 1,694 runs, 185 sweeps,
and 86,347 history rows. The history export scanned every run, contains 1,381
runs with history, and has zero export errors.

## Conclusion

Every ColoredMNIST row in Table 1 can now be given a disposition, but not every
row can be claimed as an exact CMP sweep match:

- 9 of the 15 experimental rows have a method-, dataset-, and
  representation-consistent run whose **in-domain-selected summary** matches
  the two printed values to three-decimal precision.
- 4 experimental rows have an identifiable relevant sweep, but no run in that
  sweep reproduces both printed values.
- 2 MatchDG variants have no successful run with the required representation
  in CMP.
- Random guess, ERM oracle, and theory oracle are reference rows rather than
  identified W&B sweeps.

The strongest matches use `dataset=LISAColoredMNIST`, not the repo's separate
`ColoredMNIST` class. That distinction is material and is preserved below.

## Complete row map

The logged pair is
`in_test_val_best_in_test / in_test_val_best_test`. These are the summary fields
written by `ERM.report()` after selecting the epoch with the highest in-domain
test accuracy. An arbitrary history epoch is not treated as a source match.

| Paper row | Paper in / test | CMP sweep | Best run and logged pair | Assessment |
|---|---:|---|---|---|
| ERM | .852 / .093 | [`vyjr08g2`](https://wandb.ai/inouye-lab/CMP/sweeps/vyjr08g2) | [`hogg26fk`](https://wandb.ai/inouye-lab/CMP/runs/hogg26fk): .852100 / .100400 | Relevant sweep; test differs by +.007400 |
| IRM | .799 / .118 | [`fj87kbxi`](https://wandb.ai/inouye-lab/CMP/sweeps/fj87kbxi) | [`b7txktmw`](https://wandb.ai/inouye-lab/CMP/runs/b7txktmw): .798900 / .117700 | **Reported-value match** |
| REx | .797 / .121 | [`6b8qovrf`](https://wandb.ai/inouye-lab/CMP/sweeps/6b8qovrf) | [`kybpez6i`](https://wandb.ai/inouye-lab/CMP/runs/kybpez6i): .797000 / .121200 | **Reported-value match** |
| GroupDRO | .798 / .127 | [`ednv05ar`](https://wandb.ai/inouye-lab/CMP/sweeps/ednv05ar) | [`wdo3qr89`](https://wandb.ai/inouye-lab/CMP/runs/wdo3qr89): .798000 / .126700 | **Reported-value match** |
| Fish | .798 / .118 | [`ioi8x04h`](https://wandb.ai/inouye-lab/CMP/sweeps/ioi8x04h) | [`bslvsxn1`](https://wandb.ai/inouye-lab/CMP/runs/bslvsxn1): .798400 / .118100 | **Reported-value match** |
| SWAD | .800 / .113 | [`ds25erw4`](https://wandb.ai/inouye-lab/CMP/sweeps/ds25erw4) | [`4zn1ulbq`](https://wandb.ai/inouye-lab/CMP/runs/4zn1ulbq) and `r7xby1nv`: .797600 / .113000 | Relevant sweep; in accuracy differs by -.002400; tied runs |
| LISA | .705 / .000 | [`3x4gdbd1`](https://wandb.ai/inouye-lab/CMP/sweeps/3x4gdbd1) | [`285kyk40`](https://wandb.ai/inouye-lab/CMP/runs/285kyk40): .703900 / .672300 | Methodologically relevant sweep, but a clear numerical conflict |
| MatchDG, 1NN, CNN | .698 / .361 | — | — | No successful schema-consistent CMP run |
| MatchDG, 1NN, finetune | .850 / .181 | — | — | No successful schema-consistent CMP run |
| MatchDG, random, probing | .799 / .120 | [`rma40ivl`](https://wandb.ai/inouye-lab/CMP/sweeps/rma40ivl) | [`6c17t5wp`](https://wandb.ai/inouye-lab/CMP/runs/6c17t5wp): .798600 / .119700 | **Reported-value match** |
| MatchDG, 1NN, probing | .789 / .217 | [`wm4jkkrc`](https://wandb.ai/inouye-lab/CMP/sweeps/wm4jkkrc) | [`uij14jii`](https://wandb.ai/inouye-lab/CMP/runs/uij14jii): .788200 / .217400 | Relevant sweep; close, but in accuracy rounds to .788 |
| MatchDG, clean, probing | .793 / .181 | [`8i6zxyae`](https://wandb.ai/inouye-lab/CMP/sweeps/8i6zxyae) | [`zt0o2k7c`](https://wandb.ai/inouye-lab/CMP/runs/zt0o2k7c): .792500 / .180800 | **Reported-value match** |
| GRIT, random, probing | .794 / .176 | [`b5j4s8lh`](https://wandb.ai/inouye-lab/CMP/sweeps/b5j4s8lh) | [`ojwqtd2w`](https://wandb.ai/inouye-lab/CMP/runs/ojwqtd2w): .794400 / .175600 | **Reported-value match** |
| GRIT, 1NN, probing | .736 / .649 | [`8ue2bsl0`](https://wandb.ai/inouye-lab/CMP/sweeps/8ue2bsl0) | [`e41qt7s6`](https://wandb.ai/inouye-lab/CMP/runs/e41qt7s6): .735800 / .649200 | **Reported-value match** |
| GRIT, clean, probing | .740 / .693 | [`4hf9mw1o`](https://wandb.ai/inouye-lab/CMP/sweeps/4hf9mw1o) | [`3i7strze`](https://wandb.ai/inouye-lab/CMP/runs/3i7strze): .739500 / .692500 | **Reported-value match** |
| Random guess | .500 / .500 | — | — | Analytical reference; no sweep expected |
| ERM oracle | .735 / .730 | — | — | No `split_scheme=oracle` ERM run in CMP; likely a reference calculation |
| Theory oracle | .750 / .750 | — | — | Analytical reference; no sweep expected |

## Configurations for the matched and relevant sweeps

Except for the ERM row, all listed experimental sweeps use:

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

The ERM sweep `vyjr08g2` instead uses `dataset=ColoredMNIST` and varies
`seed=[1001,...,1010]`; the closest run uses seed 1007.

| Row | Sweep name | Sweep-specific search | Selected run configuration |
|---|---|---|---|
| IRM | `ColoredMNIST-IRM-tuning` | `param1=[.01,.1,1,10,100]`, `param2=100` | `param1=10`, `param2=100` |
| REx | `ColoredMNIST-REx-tuning` | `param1=[.01,.1,1,10,100]`, `param2=100` | `param1=.1`, `param2=100` |
| GroupDRO | `ColoredMNIST-GroupDRO-sweep` | `param1=[.001,.1]` | `param1=.1` |
| Fish | `ColoredMNIST-Fish-tuning` | `param1=[.001,.01,.1]` | `param1=.1` |
| SWAD | `ColoredMNIST-SWAD-tuning` | `param1=[1.1,1.2,1.3,1.4,1.5]` | tied candidates use `param1=1.1` and `1.2` |
| LISA | `ColoredMNIST-LISA-tuning` | `param1=[0,.1,.3,.5,.7,.9,1]` | closest-in candidate uses `param1=0` |
| MatchDG random | `ColoredMNIST-MatchDG-Condition` | `param1=[.01,.1,1,10,100]`, `latent_dim=[4,8,...,32]` | `projection=conditional`, `param1=.01`, `latent_dim=4` |
| MatchDG 1NN | `ColoredMNIST-MatchDG-Nearest` | `param1=[.01,.1,1,10,100]`, `latent_dim=[4,8,...,32]` | `projection=nearest`, `param1=.1`, `latent_dim=12` |
| MatchDG clean | `ColoredMNIST-MatchDG-Oralce` | `param1=[.01,.1,1,10,100]`, `param2=1024`, `latent_dim=[4,8,...,32]` | `projection=oracle`, `param1=.01`, `param2=1024`, `latent_dim=16` |
| GRIT random | `ColoredMNIST-ECMP-Condition` | rank `param1=[2,3,4,5,6]`, `param2=512` | `projection=conditional`, rank 2, `param2=512` |
| GRIT 1NN | `ColoredMNIST-ECMP-Nearest` | rank `param1=[2,4,6,8,10]` | `projection=nearest`, rank 8 |
| GRIT clean | `ColoredMNIST-ECMP-Oracle` | rank `param1=[2,4,6,8,10,12]`, `param2=1024` | `projection=oracle`, rank 8, `param2=1024` |

The full machine-readable sweep configurations and run configurations remain in
[`wandb_sweeps.json`](wandb_sweeps.json) and
[`wandb_runs.json`](wandb_runs.json).

## Why intermediate-epoch coincidences were rejected

The complete histories contain two tempting additional numerical hits:

- In the true-`ColoredMNIST` ECMP sweep
  [`uaj2zz44`](https://wandb.ai/inouye-lab/CMP/sweeps/uaj2zz44), run
  [`04lc5hor`](https://wandb.ai/inouye-lab/CMP/runs/04lc5hor) logs
  `.740300 / .693000` at step 15. Its in-domain-selected summary is instead
  `.742000 / .677900`, so step 15 was not selected by the repo's stated
  protocol.
- In sweep `wm4jkkrc`, run
  [`ted0z0az`](https://wandb.ai/inouye-lab/CMP/runs/ted0z0az) logs
  `.789500 / .216800` at step 1. Its in-domain-selected summary is instead
  `.793600 / .159800`.

These history points show that metric-only searching can create false
attributions. They are not used as the primary mappings above.

## Remaining provenance caveats

- `LISAColoredMNIST` and `ColoredMNIST` are distinct dataset implementations.
  The nine reported-value matches cluster on the former, which is evidence that
  the paper table likely used that construction, but it should be stated rather
  than silently normalized to `ColoredMNIST`.
- The matching GRIT sweeps log `param2=512` or `1024`, while the appendix says
  256 counterfactual pairs. In the checked-in ECMP implementation, the
  `param2`-based oracle-pair subsampling lines are commented out, so the logged
  value may be inert. This is still a provenance discrepancy.
- The CMP sweep metric is configured as `test.acc_avg`, even though the paper
  labels the results as in-domain-selected. The row matching above uses the
  code's explicit in-domain-selected summary fields, not the sweep's metric.
- Numerical agreement plus schema agreement is strong evidence, not a signed
  record that the authors copied a particular run into the paper.
