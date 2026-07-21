# GRIT Table 1 W&B match report

Export: `inouye-lab/CMP`, `1694` runs and `185` sweeps.

This report separates strict schema filtering from numerical closeness. It does not silently alias `ColoredMNIST` with `LISAColoredMNIST`, treat plain `Waterbirds` as Waterbirds-CF, or invent a finetuning flag that is absent from the base parser.

## Result overview

A `reported-value match` means both logged metrics are within 0.00051 of the paper values (the tolerance implied by three-decimal reporting). It is a numerical match after the strict schema filter, not proof that no other run could share the printed values. `Closest only` is deliberately not claimed as the source run.

| Dataset       | Paper row                 | Status               | Best run   | Sweep      |     Logged in / out | Paper in / out |       L1 |
| ------------- | ------------------------- | -------------------- | ---------- | ---------- | ------------------: | -------------: | -------: |
| colored_mnist | ERM                       | Closest only         | `hogg26fk` | `vyjr08g2` | 0.852100 / 0.100400 |  0.852 / 0.093 | 0.007500 |
| colored_mnist | IRM                       | Reported-value match | `b7txktmw` | `fj87kbxi` | 0.798900 / 0.117700 |  0.799 / 0.118 | 0.000400 |
| colored_mnist | REx                       | Reported-value match | `kybpez6i` | `6b8qovrf` | 0.797000 / 0.121200 |  0.797 / 0.121 | 0.000200 |
| colored_mnist | GroupDRO                  | Reported-value match | `wdo3qr89` | `ednv05ar` | 0.798000 / 0.126700 |  0.798 / 0.127 | 0.000300 |
| colored_mnist | Fish                      | Reported-value match | `bslvsxn1` | `ioi8x04h` | 0.798400 / 0.118100 |  0.798 / 0.118 | 0.000500 |
| colored_mnist | SWAD                      | Closest only         | `4zn1ulbq` | `ds25erw4` | 0.797600 / 0.113000 |  0.800 / 0.113 | 0.002400 |
| colored_mnist | LISA                      | Closest only         | `qayxxd7u` | `fk1y7bw1` | 0.851700 / 0.100600 |  0.705 / 0.000 | 0.247300 |
| colored_mnist | MatchDG (1NN, CNN)        | Unresolved           | —          | —          |                   — |  0.698 / 0.361 |        — |
| colored_mnist | MatchDG (1NN, Finetune)   | Unresolved           | —          | —          |                   — |  0.850 / 0.181 |        — |
| colored_mnist | MatchDG (random, Probing) | Reported-value match | `6c17t5wp` | `rma40ivl` | 0.798600 / 0.119700 |  0.799 / 0.120 | 0.000700 |
| colored_mnist | MatchDG (1NN, Probing)    | Closest only         | `uij14jii` | `wm4jkkrc` | 0.788200 / 0.217400 |  0.789 / 0.217 | 0.001200 |
| colored_mnist | MatchDG (clean, Probing)  | Reported-value match | `zt0o2k7c` | `8i6zxyae` | 0.792500 / 0.180800 |  0.793 / 0.181 | 0.000700 |
| colored_mnist | GRIT (random, Probing)    | Reported-value match | `ojwqtd2w` | `b5j4s8lh` | 0.794400 / 0.175600 |  0.794 / 0.176 | 0.000800 |
| colored_mnist | GRIT (1NN, Probing)       | Reported-value match | `e41qt7s6` | `8ue2bsl0` | 0.735800 / 0.649200 |  0.736 / 0.649 | 0.000400 |
| colored_mnist | GRIT (clean, Probing)     | Reported-value match | `3i7strze` | `4hf9mw1o` | 0.739500 / 0.692500 |  0.740 / 0.693 | 0.001000 |
| colored_mnist | ERM oracle                | Unresolved           | —          | —          |                   — |  0.735 / 0.730 |        — |
| waterbirds_cf | ERM                       | Closest only         | `xud8qvn0` | `pt634tp4` | 0.778350 / 0.000000 |  0.885 / 0.781 | 0.887650 |
| waterbirds_cf | IRM                       | Closest only         | `8mp2lpxc` | `pt634tp4` | 0.775490 / 0.000000 |  0.838 / 0.707 | 0.769510 |
| waterbirds_cf | REx                       | Closest only         | `2nd62sq9` | `pt634tp4` | 0.775919 / 0.000000 |  0.891 / 0.617 | 0.732081 |
| waterbirds_cf | GroupDRO                  | Closest only         | `ott0ewoh` | `pt634tp4` | 0.675819 / 0.172932 |  0.906 / 0.684 | 0.741249 |
| waterbirds_cf | Fish                      | Closest only         | `riuytztc` | `pt634tp4` | 0.615330 / 0.191589 |  0.900 / 0.744 | 0.837082 |
| waterbirds_cf | LISA                      | Closest only         | `xidklz5f` | `mqeob79m` | 0.893179 / 0.759399 |  0.904 / 0.722 | 0.048220 |
| waterbirds_cf | MatchDG (1NN, CNN)        | Unresolved           | —          | —          |                   — |  0.970 / 0.080 |        — |
| waterbirds_cf | MatchDG (1NN, Finetune)   | Unresolved           | —          | —          |                   — |  0.920 / 0.112 |        — |
| waterbirds_cf | MatchDG (random, Probing) | Closest only         | `ypiv11e6` | `8pogorao` | 0.772773 / 0.070093 |  0.793 / 0.009 | 0.081321 |
| waterbirds_cf | MatchDG (1NN, Probing)    | Closest only         | `ro69rgv2` | `ld910a42` | 0.885028 / 0.400312 |  0.886 / 0.411 | 0.011661 |
| waterbirds_cf | MatchDG (clean, Probing)  | Closest only         | `y5q1vhgq` | `78mawi7i` | 0.893465 / 0.637072 |  0.906 / 0.536 | 0.113607 |
| waterbirds_cf | GRIT (random, Probing)    | Reported-value match | `i34a6ite` | `ep2y89o3` | 0.803947 / 0.269470 |  0.804 / 0.269 | 0.000524 |
| waterbirds_cf | GRIT (1NN, Probing)       | Closest only         | `pc0ihs74` | `jrd0pl4w` | 0.892178 / 0.521807 |  0.892 / 0.521 | 0.000985 |
| waterbirds_cf | GRIT (clean, Probing)     | Closest only         | `6e7x6weq` | `ndihtoy9` | 0.911626 / 0.643302 |  0.864 / 0.812 | 0.216324 |

Numerically matched after strict filtering: `10/30` matchable rows.

Headline conflicts:

- waterbirds_cf GRIT (clean, Probing): closest strict run 6e7x6weq in sweep ndihtoy9 logs 0.911626 / 0.643302, not 0.864 / 0.812.

## Verified paper-to-code mapping

| Paper term                                                | Code config                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| GRIT                                                      | `solver=ECMP`                                                |
| ERM / IRM / REx / GroupDRO / Fish / SWAD / LISA / MatchDG | same value in `solver`                                       |
| random pairs                                              | `projection=conditional`                                     |
| 1NN pairs                                                 | `projection=nearest`                                         |
| clean pairs                                               | `projection=oracle`                                          |
| rank r for GRIT                                           | `param1`                                                     |
| frozen CLIP linear probe                                  | `pretrained=true`, `featurizer=linear`                       |
| ColoredMNIST code values                                  | `ColoredMNIST`, `LISAColoredMNIST` (kept distinct in output) |
| Waterbirds-CF code value                                  | `CounterfactualWaterbirds`                                   |

The complete mapping, including overloaded `param1` meanings and optimizer caveats, is in `data/config_mapping.json`.

## Project inventory

Run states: `{'crashed': 188, 'failed': 181, 'finished': 1324, 'killed': 1}`.

Observed dataset config values: `{'CounterfactualWaterbirds': 148, 'LISAColorCounterfactualWaterbirdsedMNIST': 1, 'LISAColoredMNIST': 185, 'LISAColoredMNISTDataset': 2, 'RotatedMNIST': 194, 'MyLISAColoredMNIST': 4, 'Camelyon17': 76, 'ColoredMNIST': 457, 'PACS': 583, 'Waterbirds': 7, 'CounterfactualCelebA': 37}`.

Observed solver config values: `{'MatchDG': 624, 'ECMP': 406, 'matchdg': 1, 'ERM': 59, 'Fish': 87, 'SWAD': 58, 'REx': 98, 'LISA': 129, 'IRM': 127, 'GroupDRO': 105}`.

Runs with a logged or used W&B artifact whose type is exactly `dataset`: `0`.

Artifact types: `{'code': 1520, 'wandb-events': 24, 'wandb-history': 1381, 'job': 1}`.

**Dataset provenance limitation:** no run references a W&B artifact typed `dataset`; root paths in run configs cannot establish the Waterbirds-CF metadata hash or confirm the 184/56 counterfactual composition.

## colored_mnist: ERM

Paper target: in `0.852`, out `0.093`.

Strict schema filter: `20` finished runs; `20` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `vyjr08g2` |   10 | 0.851900 | 0.103040 | -0.000100 | +0.010040 | 0.010140 |
|    2 | `ovm29ejn` |   10 | 0.797710 | 0.113870 | -0.054290 | +0.020870 | 0.075160 |

<details><summary>Sweep/config aggregate 1: vyjr08g2</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `12a0rvad, dn15qp5g, f3gma30e, gpky6k3p, hogg26fk, j03judjw, ocebj8z8, oxax3nhy, tm85uzsg, yu6lt8yv`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 100,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "ERM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ERM",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["ColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    },
    "solver": {
      "values": ["ERM"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: ovm29ejn</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `1b7b1sy5, 1zgq74a0, 2on0qrhe, 86vorjp3, 9m2dp5kt, bu8rxkrp, db9z75e7, e9osross, jhowqljl, k2ho82oi`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 100,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "ERM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ERM",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    },
    "solver": {
      "values": ["ERM"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name           | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | -------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `hogg26fk` | prime-sweep-7  | `vyjr08g2` | in-domain selected scalar summary | 0.852100 / 0.100400 | +0.000100 / +0.007400 | 0.007500 |
|    2 | `tm85uzsg` | vocal-sweep-4  | `vyjr08g2` | in-domain selected scalar summary | 0.852000 / 0.100600 | -0.000000 / +0.007600 | 0.007600 |
|    3 | `oxax3nhy` | bright-sweep-9 | `vyjr08g2` | in-domain selected scalar summary | 0.852000 / 0.100600 | -0.000000 / +0.007600 | 0.007600 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: hogg26fk (prime-sweep-7)</summary>

Created: `2025-04-14T20:32:21Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/hogg26fk>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 100,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1007,
  "solver": "ERM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-14T20:17:21Z",
    "digest": "540a84fa8035e32ece9d151ca46a8402",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY1NjE0Mjk1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v176",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v176",
    "relationship": "logged",
    "size": 102885,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-14T20:17:23Z",
    "version": "v176"
  }
]
```

</details>

<details><summary>Run candidate 2: tm85uzsg (vocal-sweep-4)</summary>

Created: `2025-04-14T20:24:36Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/tm85uzsg>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 100,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1004,
  "solver": "ERM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-14T20:17:21Z",
    "digest": "540a84fa8035e32ece9d151ca46a8402",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY1NjE0Mjk1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v176",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v176",
    "relationship": "logged",
    "size": 102885,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-14T20:17:23Z",
    "version": "v176"
  }
]
```

</details>

<details><summary>Run candidate 3: oxax3nhy (bright-sweep-9)</summary>

Created: `2025-04-14T20:37:09Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/oxax3nhy>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 100,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1009,
  "solver": "ERM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-14T20:17:21Z",
    "digest": "540a84fa8035e32ece9d151ca46a8402",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY1NjE0Mjk1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v176",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v176",
    "relationship": "logged",
    "size": 102885,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-14T20:17:23Z",
    "version": "v176"
  }
]
```

</details>

## colored_mnist: IRM

Paper target: in `0.799`, out `0.118`.

Strict schema filter: `13` finished runs; `13` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `fj87kbxi` |    1 | 0.798900 | 0.117700 | -0.000100 | -0.000300 | 0.000400 |
|    2 | `fj87kbxi` |    1 | 0.798400 | 0.119700 | -0.000600 | +0.001700 | 0.002300 |
|    3 | `fj87kbxi` |    1 | 0.797700 | 0.128900 | -0.001300 | +0.010900 | 0.012200 |

**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.

<details><summary>Sweep/config aggregate 1: fj87kbxi</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `b7txktmw`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 10,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "IRM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-IRM-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "param2": {
      "values": [100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["IRM"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: fj87kbxi</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `yowlg805`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 100,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "IRM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-IRM-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "param2": {
      "values": [100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["IRM"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: fj87kbxi</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `ltuq96qt`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.01,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "IRM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-IRM-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "param2": {
      "values": [100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["IRM"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name              | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ----------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `b7txktmw` | exalted-sweep-4   | `fj87kbxi` | in-domain selected scalar summary | 0.798900 / 0.117700 | -0.000100 / -0.000300 | 0.000400 |
|    2 | `yowlg805` | glamorous-sweep-5 | `fj87kbxi` | in-domain selected scalar summary | 0.798400 / 0.119700 | -0.000600 / +0.001700 | 0.002300 |
|    3 | `ltuq96qt` | sweepy-sweep-1    | `fj87kbxi` | in-domain selected scalar summary | 0.797700 / 0.128900 | -0.001300 / +0.010900 | 0.012200 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: b7txktmw (exalted-sweep-4)</summary>

Created: `2025-04-23T16:04:57Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/b7txktmw>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 10,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "IRM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 2: yowlg805 (glamorous-sweep-5)</summary>

Created: `2025-04-23T16:07:24Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/yowlg805>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 100,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "IRM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 3: ltuq96qt (sweepy-sweep-1)</summary>

Created: `2025-04-23T15:57:38Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/ltuq96qt>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.01,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "IRM",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T15:57:55Z",
    "digest": "57eb717b009ddc0f767f7a47596f38a6",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA3MzI1MQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v292",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v292",
    "relationship": "logged",
    "size": 160811,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T15:57:57Z",
    "version": "v292"
  }
]
```

</details>

## colored_mnist: REx

Paper target: in `0.797`, out `0.121`.

Strict schema filter: `13` finished runs; `13` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `6b8qovrf` |    1 | 0.797000 | 0.121200 | -0.000000 | +0.000200 | 0.000200 |
|    2 | `6b8qovrf` |    1 | 0.796700 | 0.117300 | -0.000300 | -0.003700 | 0.004000 |
|    3 | `6b8qovrf` |    1 | 0.795300 | 0.124600 | -0.001700 | +0.003600 | 0.005300 |

<details><summary>Sweep/config aggregate 1: 6b8qovrf</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `kybpez6i`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "REx",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-REx-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "param2": {
      "values": [100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["REx"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: 6b8qovrf</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `j9lvk83k`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.01,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "REx",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-REx-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "param2": {
      "values": [100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["REx"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: 6b8qovrf</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `n6iai4xh`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 10,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "REx",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-REx-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "param2": {
      "values": [100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["REx"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name          | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `kybpez6i` | drawn-sweep-2 | `6b8qovrf` | in-domain selected scalar summary | 0.797000 / 0.121200 | -0.000000 / +0.000200 | 0.000200 |
|    2 | `j9lvk83k` | sleek-sweep-1 | `6b8qovrf` | in-domain selected scalar summary | 0.796700 / 0.117300 | -0.000300 / -0.003700 | 0.004000 |
|    3 | `n6iai4xh` | wise-sweep-4  | `6b8qovrf` | in-domain selected scalar summary | 0.795300 / 0.124600 | -0.001700 / +0.003600 | 0.005300 |

<details><summary>Run candidate 1: kybpez6i (drawn-sweep-2)</summary>

Created: `2025-04-23T16:00:43Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/kybpez6i>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "REx",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

<details><summary>Run candidate 2: j9lvk83k (sleek-sweep-1)</summary>

Created: `2025-04-23T15:58:26Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/j9lvk83k>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.01,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "REx",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

<details><summary>Run candidate 3: n6iai4xh (wise-sweep-4)</summary>

Created: `2025-04-23T16:05:37Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/n6iai4xh>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 10,
  "param2": 100,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "REx",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

## colored_mnist: GroupDRO

Paper target: in `0.798`, out `0.127`.

Strict schema filter: `26` finished runs; `25` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `ednv05ar` |    1 | 0.798000 | 0.126700 | -0.000000 | -0.000300 | 0.000300 |
|    2 | `ednv05ar` |    1 | 0.797500 | 0.127800 | -0.000500 | +0.000800 | 0.001300 |
|    3 | `651qfh0l` |    1 | 0.848900 | 0.122300 | +0.050900 | -0.004700 | 0.055600 |

**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.

<details><summary>Sweep/config aggregate 1: ednv05ar</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `wdo3qr89`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "GroupDRO",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-GroupDRO-sweep",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.001, 0.1]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["GroupDRO"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: ednv05ar</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `qfekvjth`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.001,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "GroupDRO",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-GroupDRO-sweep",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.001, 0.1]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["GroupDRO"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: 651qfh0l</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `06w9r0uz`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "GroupDRO",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-GroupDRO-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["ColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.001, 0.01, 0.1]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["GroupDRO"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name          | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `wdo3qr89` | daily-sweep-2 | `ednv05ar` | in-domain selected scalar summary | 0.798000 / 0.126700 | -0.000000 / -0.000300 | 0.000300 |
|    2 | `qfekvjth` | leafy-sweep-1 | `ednv05ar` | in-domain selected scalar summary | 0.797500 / 0.127800 | -0.000500 / +0.000800 | 0.001300 |
|    3 | `jpx5x3ne` | cool-sweep-18 | `t44zi3ge` | in-domain selected scalar summary | 0.847900 / 0.130700 | +0.049900 / +0.003700 | 0.053600 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: wdo3qr89 (daily-sweep-2)</summary>

Created: `2025-04-23T16:01:29Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/wdo3qr89>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "GroupDRO",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 2: qfekvjth (leafy-sweep-1)</summary>

Created: `2025-04-23T15:58:53Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/qfekvjth>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.001,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "GroupDRO",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

<details><summary>Run candidate 3: jpx5x3ne (cool-sweep-18)</summary>

Created: `2025-04-15T13:35:26Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/jpx5x3ne>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1008,
  "solver": "GroupDRO",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-15T13:20:31Z",
    "digest": "80ebbd2f08221c1daa132bbf7a835660",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY1ODIzMzY5OQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v181",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v181",
    "relationship": "logged",
    "size": 104246,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-15T13:20:33Z",
    "version": "v181"
  }
]
```

</details>

## colored_mnist: Fish

Paper target: in `0.798`, out `0.118`.

Strict schema filter: `9` finished runs; `9` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `ioi8x04h` |    1 | 0.798400 | 0.118100 | +0.000400 | +0.000100 | 0.000500 |
|    2 | `ioi8x04h` |    1 | 0.797300 | 0.105000 | -0.000700 | -0.013000 | 0.013700 |
|    3 | `ioi8x04h` |    1 | 0.761200 | 0.135700 | -0.036800 | +0.017700 | 0.054500 |

<details><summary>Sweep/config aggregate 1: ioi8x04h</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `bslvsxn1`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "Fish",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-Fish-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.001, 0.01, 0.1]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["Fish"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: ioi8x04h</summary>

Metric source: `latest nested split summary`; run IDs: `y09lqsb5`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.01,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "Fish",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-Fish-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.001, 0.01, 0.1]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["Fish"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: ioi8x04h</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `pm7navm0`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.001,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "Fish",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-Fish-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.001, 0.01, 0.1]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["Fish"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name          | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `bslvsxn1` | jolly-sweep-3 | `ioi8x04h` | in-domain selected scalar summary | 0.798400 / 0.118100 | +0.000400 / +0.000100 | 0.000500 |
|    2 | `y09lqsb5` | eager-sweep-2 | `ioi8x04h` | latest nested split summary       | 0.797300 / 0.105000 | -0.000700 / -0.013000 | 0.013700 |
|    3 | `pm7navm0` | comfy-sweep-1 | `ioi8x04h` | in-domain selected scalar summary | 0.761200 / 0.135700 | -0.036800 / +0.017700 | 0.054500 |

<details><summary>Run candidate 1: bslvsxn1 (jolly-sweep-3)</summary>

Created: `2025-04-23T16:04:46Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/bslvsxn1>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "Fish",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 2: y09lqsb5 (eager-sweep-2)</summary>

Created: `2025-04-23T16:02:04Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/y09lqsb5>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.01,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "Fish",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 3: pm7navm0 (comfy-sweep-1)</summary>

Created: `2025-04-23T15:59:22Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/pm7navm0>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.001,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "Fish",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

## colored_mnist: SWAD

Paper target: in `0.800`, out `0.113`.

Strict schema filter: `7` finished runs; `7` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `ds25erw4` |    1 | 0.797600 | 0.113000 | -0.002400 | -0.000000 | 0.002400 |
|    2 | `ds25erw4` |    1 | 0.797600 | 0.113000 | -0.002400 | -0.000000 | 0.002400 |
|    3 | `0e9qy2wp` |    1 | 0.851800 | 0.100800 | +0.051800 | -0.012200 | 0.064000 |

**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.

<details><summary>Sweep/config aggregate 1: ds25erw4</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `4zn1ulbq`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 1.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "SWAD",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-SWAD-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [1.1, 1.2, 1.3, 1.4, 1.5]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["SWAD"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: ds25erw4</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `r7xby1nv`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 1.2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "SWAD",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-SWAD-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [1.1, 1.2, 1.3, 1.4, 1.5]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["SWAD"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: 0e9qy2wp</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `1q87w4zz`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 1.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "SWAD",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-SWAD-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["ColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [1.1, 1.2, 1.3, 1.4, 1.5]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["SWAD"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name           | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | -------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `4zn1ulbq` | wild-sweep-1   | `ds25erw4` | in-domain selected scalar summary | 0.797600 / 0.113000 | -0.002400 / -0.000000 | 0.002400 |
|    2 | `r7xby1nv` | fresh-sweep-2  | `ds25erw4` | in-domain selected scalar summary | 0.797600 / 0.113000 | -0.002400 / -0.000000 | 0.002400 |
|    3 | `1q87w4zz` | fallen-sweep-1 | `0e9qy2wp` | in-domain selected scalar summary | 0.851800 / 0.100800 | +0.051800 / -0.012200 | 0.064000 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: 4zn1ulbq (wild-sweep-1)</summary>

Created: `2025-04-23T16:01:45Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/4zn1ulbq>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 1.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "SWAD",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 2: r7xby1nv (fresh-sweep-2)</summary>

Created: `2025-04-23T16:21:38Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/r7xby1nv>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 1.2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "SWAD",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:21:46Z",
    "digest": "b934a5f44f1308edc274404bb687616c",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzExNzQzNQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v296",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v296",
    "relationship": "logged",
    "size": 161326,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:21:48Z",
    "version": "v296"
  }
]
```

</details>

<details><summary>Run candidate 3: 1q87w4zz (fallen-sweep-1)</summary>

Created: `2025-04-22T16:08:00Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/1q87w4zz>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 1.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "SWAD",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-22T16:08:00Z",
    "digest": "8d4fc5c934ba97ead53495ba9542b90a",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NDQzMzYyNQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v258",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v258",
    "relationship": "logged",
    "size": 137747,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-22T16:08:03Z",
    "version": "v258"
  }
]
```

</details>

## colored_mnist: LISA

Paper target: in `0.705`, out `0.000`.

Strict schema filter: `14` finished runs; `14` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `fk1y7bw1` |    1 | 0.851700 | 0.100600 | +0.146700 | +0.100600 | 0.247300 |
|    2 | `fk1y7bw1` |    1 | 0.851700 | 0.100600 | +0.146700 | +0.100600 | 0.247300 |
|    3 | `fk1y7bw1` |    1 | 0.851700 | 0.100600 | +0.146700 | +0.100600 | 0.247300 |

**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.

<details><summary>Sweep/config aggregate 1: fk1y7bw1</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `qayxxd7u`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-LISA-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["ColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["LISA"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: fk1y7bw1</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `7snvbrfl`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.5,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-LISA-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["ColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["LISA"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: fk1y7bw1</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `cihhavbk`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.7,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-LISA-tuning",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["ColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["LISA"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name            | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | --------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `qayxxd7u` | smooth-sweep-2  | `fk1y7bw1` | in-domain selected scalar summary | 0.851700 / 0.100600 | +0.146700 / +0.100600 | 0.247300 |
|    2 | `7snvbrfl` | stellar-sweep-4 | `fk1y7bw1` | in-domain selected scalar summary | 0.851700 / 0.100600 | +0.146700 / +0.100600 | 0.247300 |
|    3 | `cihhavbk` | clean-sweep-5   | `fk1y7bw1` | in-domain selected scalar summary | 0.851700 / 0.100600 | +0.146700 / +0.100600 | 0.247300 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: qayxxd7u (smooth-sweep-2)</summary>

Created: `2025-04-22T16:12:46Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/qayxxd7u>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-22T16:08:00Z",
    "digest": "8d4fc5c934ba97ead53495ba9542b90a",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NDQzMzYyNQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v258",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v258",
    "relationship": "logged",
    "size": 137747,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-22T16:08:03Z",
    "version": "v258"
  }
]
```

</details>

<details><summary>Run candidate 2: 7snvbrfl (stellar-sweep-4)</summary>

Created: `2025-04-22T16:22:06Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/7snvbrfl>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.5,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-22T16:08:00Z",
    "digest": "8d4fc5c934ba97ead53495ba9542b90a",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NDQzMzYyNQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v258",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v258",
    "relationship": "logged",
    "size": 137747,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-22T16:08:03Z",
    "version": "v258"
  }
]
```

</details>

<details><summary>Run candidate 3: cihhavbk (clean-sweep-5)</summary>

Created: `2025-04-22T16:26:40Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/cihhavbk>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.7,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-22T16:08:00Z",
    "digest": "8d4fc5c934ba97ead53495ba9542b90a",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NDQzMzYyNQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v258",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v258",
    "relationship": "logged",
    "size": 137747,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-22T16:08:03Z",
    "version": "v258"
  }
]
```

</details>

## colored_mnist: MatchDG (1NN, CNN)

Paper target: in `0.698`, out `0.361`.

Strict schema filter: `0` finished runs; `0` expose a usable summary metric pair.

No candidate can be selected from the exported summaries. This is reported as an unresolved row, not guessed.

## colored_mnist: MatchDG (1NN, Finetune)

Paper target: in `0.850`, out `0.181`.

Strict schema filter: `0` finished runs; `0` expose a usable summary metric pair.

No candidate can be selected from the exported summaries. This is reported as an unresolved row, not guessed.

## colored_mnist: MatchDG (random, Probing)

Paper target: in `0.799`, out `0.120`.

Strict schema filter: `69` finished runs; `69` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `rma40ivl` |    1 | 0.798600 | 0.119700 | -0.000400 | -0.000300 | 0.000700 |
|    2 | `rma40ivl` |    1 | 0.795600 | 0.123600 | -0.003400 | +0.003600 | 0.007000 |
|    3 | `rma40ivl` |    1 | 0.796500 | 0.124600 | -0.002500 | +0.004600 | 0.007100 |

<details><summary>Sweep/config aggregate 1: rma40ivl</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `6c17t5wp`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 4,
  "lr": 0.001,
  "param1": 0.01,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-MatchDG-Condition",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [4, 8, 12, 16, 20, 24, 28, 32]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: rma40ivl</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `d3rcgasm`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 12,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-MatchDG-Condition",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [4, 8, 12, 16, 20, 24, 28, 32]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: rma40ivl</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `dhst4xvn`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 8,
  "lr": 0.001,
  "param1": 1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-MatchDG-Condition",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [4, 8, 12, 16, 20, 24, 28, 32]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name           | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | -------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `6c17t5wp` | whole-sweep-1  | `rma40ivl` | in-domain selected scalar summary | 0.798600 / 0.119700 | -0.000400 / -0.000300 | 0.000700 |
|    2 | `d3rcgasm` | still-sweep-12 | `rma40ivl` | in-domain selected scalar summary | 0.795600 / 0.123600 | -0.003400 / +0.003600 | 0.007000 |
|    3 | `dhst4xvn` | dandy-sweep-8  | `rma40ivl` | in-domain selected scalar summary | 0.796500 / 0.124600 | -0.002500 / +0.004600 | 0.007100 |

<details><summary>Run candidate 1: 6c17t5wp (whole-sweep-1)</summary>

Created: `2025-04-23T16:15:20Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/6c17t5wp>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 4,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.01,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 2: d3rcgasm (still-sweep-12)</summary>

Created: `2025-04-23T16:43:01Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/d3rcgasm>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 12,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

<details><summary>Run candidate 3: dhst4xvn (dandy-sweep-8)</summary>

Created: `2025-04-23T16:33:34Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/dhst4xvn>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 8,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:27:27Z",
    "digest": "a8254fde575072df6d40ecadc0b7df18",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzEyNzk3NA==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v300",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v300",
    "relationship": "logged",
    "size": 161647,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:27:29Z",
    "version": "v300"
  }
]
```

</details>

## colored_mnist: MatchDG (1NN, Probing)

Paper target: in `0.789`, out `0.217`.

Strict schema filter: `70` finished runs; `70` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `wm4jkkrc` |    1 | 0.788200 | 0.217400 | -0.000800 | +0.000400 | 0.001200 |
|    2 | `wm4jkkrc` |    1 | 0.785800 | 0.220800 | -0.003200 | +0.003800 | 0.007000 |
|    3 | `wm4jkkrc` |    1 | 0.786300 | 0.209100 | -0.002700 | -0.007900 | 0.010600 |

<details><summary>Sweep/config aggregate 1: wm4jkkrc</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `uij14jii`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 12,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-MatchDG-Nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [4, 8, 12, 16, 20, 24, 28, 32]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: wm4jkkrc</summary>

Metric source: `latest nested split summary`; run IDs: `gp4zf787`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 16,
  "lr": 0.001,
  "param1": 0.01,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-MatchDG-Nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [4, 8, 12, 16, 20, 24, 28, 32]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: wm4jkkrc</summary>

Metric source: `latest nested split summary`; run IDs: `fd7v7cx2`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 12,
  "lr": 0.001,
  "param1": 0.01,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-MatchDG-Nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [4, 8, 12, 16, 20, 24, 28, 32]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name              | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ----------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `uij14jii` | fanciful-sweep-12 | `wm4jkkrc` | in-domain selected scalar summary | 0.788200 / 0.217400 | -0.000800 / +0.000400 | 0.001200 |
|    2 | `gp4zf787` | rich-sweep-16     | `wm4jkkrc` | latest nested split summary       | 0.785800 / 0.220800 | -0.003200 / +0.003800 | 0.007000 |
|    3 | `fd7v7cx2` | revived-sweep-11  | `wm4jkkrc` | latest nested split summary       | 0.786300 / 0.209100 | -0.002700 / -0.007900 | 0.010600 |

<details><summary>Run candidate 1: uij14jii (fanciful-sweep-12)</summary>

Created: `2025-04-23T16:43:01Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/uij14jii>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 12,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

<details><summary>Run candidate 2: gp4zf787 (rich-sweep-16)</summary>

Created: `2025-04-23T16:52:33Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/gp4zf787>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 16,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.01,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:48:39Z",
    "digest": "e569fa6c183ddbfaa3f38a17960d3fb8",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzE2OTM4Ng==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v310",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v310",
    "relationship": "logged",
    "size": 161630,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:48:41Z",
    "version": "v310"
  }
]
```

</details>

<details><summary>Run candidate 3: fd7v7cx2 (revived-sweep-11)</summary>

Created: `2025-04-23T16:40:39Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/fd7v7cx2>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 12,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.01,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

## colored_mnist: MatchDG (clean, Probing)

Paper target: in `0.793`, out `0.181`.

Strict schema filter: `86` finished runs; `86` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `8i6zxyae` |    1 | 0.792500 | 0.180800 | -0.000500 | -0.000200 | 0.000700 |
|    2 | `8i6zxyae` |    1 | 0.789900 | 0.182000 | -0.003100 | +0.001000 | 0.004100 |
|    3 | `8i6zxyae` |    1 | 0.789300 | 0.182200 | -0.003700 | +0.001200 | 0.004900 |

<details><summary>Sweep/config aggregate 1: 8i6zxyae</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `zt0o2k7c`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 16,
  "lr": 0.001,
  "param1": 0.01,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-MatchDG-Oralce",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [4, 8, 12, 16, 20, 24, 28, 32]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "param2": {
      "values": [1024]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: 8i6zxyae</summary>

Metric source: `latest nested split summary`; run IDs: `euq57rmn`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 12,
  "lr": 0.001,
  "param1": 0.01,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-MatchDG-Oralce",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [4, 8, 12, 16, 20, 24, 28, 32]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "param2": {
      "values": [1024]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: 8i6zxyae</summary>

Metric source: `latest nested split summary`; run IDs: `jxmoom6n`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 28,
  "lr": 0.001,
  "param1": 0.01,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-MatchDG-Oralce",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [4, 8, 12, 16, 20, 24, 28, 32]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.01, 0.1, 1, 10, 100]
    },
    "param2": {
      "values": [1024]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name              | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ----------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `zt0o2k7c` | mild-sweep-16     | `8i6zxyae` | in-domain selected scalar summary | 0.792500 / 0.180800 | -0.000500 / -0.000200 | 0.000700 |
|    2 | `euq57rmn` | splendid-sweep-11 | `8i6zxyae` | latest nested split summary       | 0.789900 / 0.182000 | -0.003100 / +0.001000 | 0.004100 |
|    3 | `jxmoom6n` | denim-sweep-31    | `8i6zxyae` | latest nested split summary       | 0.789300 / 0.182200 | -0.003700 / +0.001200 | 0.004900 |

<details><summary>Run candidate 1: zt0o2k7c (mild-sweep-16)</summary>

Created: `2025-04-23T16:26:11Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/zt0o2k7c>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 16,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.01,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:26:17Z",
    "digest": "778bc84fae7a427d2e0f8b1be18fc98b",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzEyNTg3Nw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v299",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v299",
    "relationship": "logged",
    "size": 161644,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:26:20Z",
    "version": "v299"
  }
]
```

</details>

<details><summary>Run candidate 2: euq57rmn (splendid-sweep-11)</summary>

Created: `2025-04-23T16:12:15Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/euq57rmn>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 12,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.01,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 3: jxmoom6n (denim-sweep-31)</summary>

Created: `2025-04-23T17:02:38Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/jxmoom6n>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 28,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.01,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:00:22Z",
    "digest": "c9c53e7341d005d05fd7592bde5c45a8",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzE5MDkyNg==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v312",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v312",
    "relationship": "logged",
    "size": 162313,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:00:24Z",
    "version": "v312"
  }
]
```

</details>

## colored_mnist: GRIT (random, Probing)

Paper target: in `0.794`, out `0.176`.

Strict schema filter: `64` finished runs; `64` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `b5j4s8lh` |    1 | 0.794400 | 0.175600 | +0.000400 | -0.000400 | 0.000800 |
|    2 | `fzbfr92t` |    1 | 0.832700 | 0.198900 | +0.038700 | +0.022900 | 0.061600 |
|    3 | `jmfp89it` |    1 | 0.832700 | 0.198900 | +0.038700 | +0.022900 | 0.061600 |

<details><summary>Sweep/config aggregate 1: b5j4s8lh</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `ojwqtd2w`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 2,
  "param2": 512,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ECMP-Condition",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 3, 4, 5, 6]
    },
    "param2": {
      "values": [512]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: fzbfr92t</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `netfe2u6`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 2,
  "param2": 512,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ECMP-Condition",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["ColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 3, 4, 5, 6]
    },
    "param2": {
      "values": [512]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: jmfp89it</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `f4lvcfja`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 2,
  "param2": 512,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ECMP-Condition",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["ColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 3, 4, 5, 6]
    },
    "param2": {
      "values": [512]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name            | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | --------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `ojwqtd2w` | glowing-sweep-1 | `b5j4s8lh` | in-domain selected scalar summary | 0.794400 / 0.175600 | +0.000400 / -0.000400 | 0.000800 |
|    2 | `netfe2u6` | crisp-sweep-1   | `fzbfr92t` | in-domain selected scalar summary | 0.832700 / 0.198900 | +0.038700 / +0.022900 | 0.061600 |
|    3 | `f4lvcfja` | honest-sweep-1  | `jmfp89it` | in-domain selected scalar summary | 0.832700 / 0.198900 | +0.038700 / +0.022900 | 0.061600 |

<details><summary>Run candidate 1: ojwqtd2w (glowing-sweep-1)</summary>

Created: `2025-04-23T16:13:24Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/ojwqtd2w>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", "param2=512 differs from the paper's 256 counterfactual pairs; checked-in subsampling is commented out", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 2,
  "param2": 512,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 2: netfe2u6 (crisp-sweep-1)</summary>

Created: `2025-04-21T12:21:37Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/netfe2u6>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", "param2=512 differs from the paper's 256 counterfactual pairs; checked-in subsampling is commented out", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 2,
  "param2": 512,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-21T12:21:52Z",
    "digest": "b9dc437e3abcb6b22398eb29a973c120",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3MTQ4OTM2NA==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v229",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v229",
    "relationship": "logged",
    "size": 120123,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-21T12:21:54Z",
    "version": "v229"
  }
]
```

</details>

<details><summary>Run candidate 3: f4lvcfja (honest-sweep-1)</summary>

Created: `2025-04-22T14:48:30Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/f4lvcfja>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", "param2=512 differs from the paper's 256 counterfactual pairs; checked-in subsampling is commented out", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 2,
  "param2": 512,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-22T13:50:07Z",
    "digest": "c43e029b75ad5c0a43a6589c9c6b56a8",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NDEyODUxOA==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v250",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v250",
    "relationship": "logged",
    "size": 132125,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-22T13:50:09Z",
    "version": "v250"
  }
]
```

</details>

## colored_mnist: GRIT (1NN, Probing)

Paper target: in `0.736`, out `0.649`.

Strict schema filter: `31` finished runs; `31` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `8ue2bsl0` |    1 | 0.735800 | 0.649200 | -0.000200 | +0.000200 | 0.000400 |
|    2 | `8ue2bsl0` |    1 | 0.731400 | 0.645700 | -0.004600 | -0.003300 | 0.007900 |
|    3 | `8ue2bsl0` |    1 | 0.723100 | 0.645900 | -0.012900 | -0.003100 | 0.016000 |

<details><summary>Sweep/config aggregate 1: 8ue2bsl0</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `e41qt7s6`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 8,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ECMP-Nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 4, 6, 8, 10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: 8ue2bsl0</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `fv51at6p`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 4,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ECMP-Nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 4, 6, 8, 10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: 8ue2bsl0</summary>

Metric source: `latest nested split summary`; run IDs: `lnoo26pt`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ECMP-Nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 4, 6, 8, 10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name             | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ---------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `e41qt7s6` | drawn-sweep-4    | `8ue2bsl0` | in-domain selected scalar summary | 0.735800 / 0.649200 | -0.000200 / +0.000200 | 0.000400 |
|    2 | `fv51at6p` | dazzling-sweep-2 | `8ue2bsl0` | in-domain selected scalar summary | 0.731400 / 0.645700 | -0.004600 / -0.003300 | 0.007900 |
|    3 | `lnoo26pt` | amber-sweep-1    | `8ue2bsl0` | latest nested split summary       | 0.723100 / 0.645900 | -0.012900 / -0.003100 | 0.016000 |

<details><summary>Run candidate 1: e41qt7s6 (drawn-sweep-4)</summary>

Created: `2025-04-23T15:54:48Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/e41qt7s6>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", "param2=0 differs from the paper's 256 counterfactual pairs; checked-in subsampling is commented out", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 8,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

<details><summary>Run candidate 2: fv51at6p (dazzling-sweep-2)</summary>

Created: `2025-04-23T15:50:14Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/fv51at6p>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", "param2=0 differs from the paper's 256 counterfactual pairs; checked-in subsampling is commented out", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 4,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

<details><summary>Run candidate 3: lnoo26pt (amber-sweep-1)</summary>

Created: `2025-04-23T15:48:04Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/lnoo26pt>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", "param2=0 differs from the paper's 256 counterfactual pairs; checked-in subsampling is commented out", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

## colored_mnist: GRIT (clean, Probing)

Paper target: in `0.740`, out `0.693`.

Strict schema filter: `90` finished runs; `83` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `4hf9mw1o` |    1 | 0.739500 | 0.692500 | -0.000500 | -0.000500 | 0.001000 |
|    2 | `4hf9mw1o` |    1 | 0.738200 | 0.694500 | -0.001800 | +0.001500 | 0.003300 |
|    3 | `4x7yoxg7` |    1 | 0.736900 | 0.691300 | -0.003100 | -0.001700 | 0.004800 |

<details><summary>Sweep/config aggregate 1: 4hf9mw1o</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `3i7strze`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 8,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ECMP-Oracle",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 4, 6, 8, 10, 12]
    },
    "param2": {
      "values": [1024]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: 4hf9mw1o</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `addw0d70`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 2,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ECMP-Oracle",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["LISAColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 4, 6, 8, 10, 12]
    },
    "param2": {
      "values": [1024]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: 4x7yoxg7</summary>

Metric source: `in-domain selected scalar summary`; run IDs: `655a6pe9`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 3,
  "param2": 512,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "ColoredMNIST-ECMP-Oracle",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["ColoredMNIST"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    },
    "param2": {
      "values": [512]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name            | Sweep      | Metric source                     | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | --------------- | ---------- | --------------------------------- | ------------------- | --------------------- | -------: |
|    1 | `3i7strze` | warm-sweep-4    | `4hf9mw1o` | in-domain selected scalar summary | 0.739500 / 0.692500 | -0.000500 / -0.000500 | 0.001000 |
|    2 | `addw0d70` | helpful-sweep-1 | `4hf9mw1o` | in-domain selected scalar summary | 0.738200 / 0.694500 | -0.001800 / +0.001500 | 0.003300 |
|    3 | `655a6pe9` | golden-sweep-4  | `4x7yoxg7` | in-domain selected scalar summary | 0.736900 / 0.691300 | -0.003100 / -0.001700 | 0.004800 |

<details><summary>Run candidate 1: 3i7strze (warm-sweep-4)</summary>

Created: `2025-04-23T16:01:26Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/3i7strze>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", "param2=1024 differs from the paper's 256 counterfactual pairs; checked-in subsampling is commented out", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 8,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:01:44Z",
    "digest": "98b8452fd86fac76852c8b32de6bd610",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzA4MDM3NQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v294",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v294",
    "relationship": "logged",
    "size": 161444,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:01:46Z",
    "version": "v294"
  }
]
```

</details>

<details><summary>Run candidate 2: addw0d70 (helpful-sweep-1)</summary>

Created: `2025-04-23T15:54:28Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/addw0d70>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'LISAColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", "param2=1024 differs from the paper's 256 counterfactual pairs; checked-in subsampling is commented out", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "LISAColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 2,
  "param2": 1024,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

<details><summary>Run candidate 3: 655a6pe9 (golden-sweep-4)</summary>

Created: `2025-04-22T14:57:31Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/655a6pe9>.

Appendix-config mismatches: `none`.

Provenance warnings: `["dataset config is 'ColoredMNIST'; ColoredMNIST and LISAColoredMNIST are distinct checked-in constructions", "param2=512 differs from the paper's 256 counterfactual pairs; checked-in subsampling is commented out", 'no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "ColoredMNIST",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 3,
  "param2": 512,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-22T14:49:43Z",
    "digest": "5ed4d26bf07f4960e034d3a3fb75c374",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NDI2MTU2MQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v251",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v251",
    "relationship": "logged",
    "size": 132121,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-22T14:49:45Z",
    "version": "v251"
  }
]
```

</details>

## colored_mnist: random guess

Paper target: in `0.500`, out `0.500`.

This is a paper reference row rather than an implemented/logged method; no W&B run is selected.

## colored_mnist: ERM oracle

Paper target: in `0.735`, out `0.730`.

Strict schema filter: `0` finished runs; `0` expose a usable summary metric pair.

No candidate can be selected from the exported summaries. This is reported as an unresolved row, not guessed.

## colored_mnist: theory oracle

Paper target: in `0.750`, out `0.750`.

This is a paper reference row rather than an implemented/logged method; no W&B run is selected.

## waterbirds_cf: ERM

Paper target: in `0.885`, out `0.781`.

Strict schema filter: `2` finished runs; `2` expose a usable summary metric pair.
No metric-bearing candidate matched every logged appendix hyperparameter; ranking uses the strict schema-filtered fallback set.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `b8i9pme1` |    1 | 0.778350 | 0.000000 | -0.106650 | -0.781000 | 0.887650 |
|    2 | `pt634tp4` |    1 | 0.778350 | 0.000000 | -0.106650 | -0.781000 | 0.887650 |

**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.

<details><summary>Sweep/config aggregate 1: b8i9pme1</summary>

Metric source: `latest nested test summary`; run IDs: `xue7kbwh`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "ERM",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: pt634tp4</summary>

Metric source: `latest nested test summary`; run IDs: `xud8qvn0`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "ERM",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name           | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | -------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `xud8qvn0` | ruby-sweep-1   | `pt634tp4` | latest nested test summary | 0.778350 / 0.000000 | -0.106650 / -0.781000 | 0.887650 |
|    2 | `xue7kbwh` | sweepy-sweep-1 | `b8i9pme1` | latest nested test summary | 0.778350 / 0.000000 | -0.106650 / -0.781000 | 0.887650 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: xud8qvn0 (ruby-sweep-1)</summary>

Created: `2025-04-07T23:04:39Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/xud8qvn0>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ERM",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:04:54Z",
    "digest": "8077dcb4ffed9548fac6ff6c4cde6530",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjkzNDY2Mg==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v7",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v7",
    "relationship": "logged",
    "size": 67435,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:04:56Z",
    "version": "v7"
  }
]
```

</details>

<details><summary>Run candidate 2: xue7kbwh (sweepy-sweep-1)</summary>

Created: `2025-04-07T23:13:09Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/xue7kbwh>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ERM",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:13:24Z",
    "digest": "45b83efc8a0483cb6d37454b1f7f4dfb",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjk0OTAxOQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v9",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v9",
    "relationship": "logged",
    "size": 67338,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:13:25Z",
    "version": "v9"
  }
]
```

</details>

## waterbirds_cf: IRM

Paper target: in `0.838`, out `0.707`.

Strict schema filter: `2` finished runs; `2` expose a usable summary metric pair.
No metric-bearing candidate matched every logged appendix hyperparameter; ranking uses the strict schema-filtered fallback set.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `b8i9pme1` |    1 | 0.775490 | 0.000000 | -0.062510 | -0.707000 | 0.769510 |
|    2 | `pt634tp4` |    1 | 0.775490 | 0.000000 | -0.062510 | -0.707000 | 0.769510 |

**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.

<details><summary>Sweep/config aggregate 1: b8i9pme1</summary>

Metric source: `latest nested test summary`; run IDs: `79x8oryn`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "IRM",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: pt634tp4</summary>

Metric source: `latest nested test summary`; run IDs: `8mp2lpxc`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "IRM",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name            | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | --------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `8mp2lpxc` | glad-sweep-3    | `pt634tp4` | latest nested test summary | 0.775490 / 0.000000 | -0.062510 / -0.707000 | 0.769510 |
|    2 | `79x8oryn` | eternal-sweep-3 | `b8i9pme1` | latest nested test summary | 0.775490 / 0.000000 | -0.062510 / -0.707000 | 0.769510 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: 8mp2lpxc (glad-sweep-3)</summary>

Created: `2025-04-07T23:05:34Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/8mp2lpxc>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "IRM",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:04:54Z",
    "digest": "8077dcb4ffed9548fac6ff6c4cde6530",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjkzNDY2Mg==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v7",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v7",
    "relationship": "logged",
    "size": 67435,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:04:56Z",
    "version": "v7"
  }
]
```

</details>

<details><summary>Run candidate 2: 79x8oryn (eternal-sweep-3)</summary>

Created: `2025-04-07T23:14:03Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/79x8oryn>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "IRM",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:13:24Z",
    "digest": "45b83efc8a0483cb6d37454b1f7f4dfb",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjk0OTAxOQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v9",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v9",
    "relationship": "logged",
    "size": 67338,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:13:25Z",
    "version": "v9"
  }
]
```

</details>

## waterbirds_cf: REx

Paper target: in `0.891`, out `0.617`.

Strict schema filter: `2` finished runs; `2` expose a usable summary metric pair.
No metric-bearing candidate matched every logged appendix hyperparameter; ranking uses the strict schema-filtered fallback set.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `b8i9pme1` |    1 | 0.775919 | 0.000000 | -0.115081 | -0.617000 | 0.732081 |
|    2 | `pt634tp4` |    1 | 0.775919 | 0.000000 | -0.115081 | -0.617000 | 0.732081 |

**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.

<details><summary>Sweep/config aggregate 1: b8i9pme1</summary>

Metric source: `latest nested test summary`; run IDs: `8c74sww3`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "REx",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: pt634tp4</summary>

Metric source: `latest nested test summary`; run IDs: `2nd62sq9`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "REx",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name              | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ----------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `2nd62sq9` | different-sweep-5 | `pt634tp4` | latest nested test summary | 0.775919 / 0.000000 | -0.115081 / -0.617000 | 0.732081 |
|    2 | `8c74sww3` | chocolate-sweep-5 | `b8i9pme1` | latest nested test summary | 0.775919 / 0.000000 | -0.115081 / -0.617000 | 0.732081 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: 2nd62sq9 (different-sweep-5)</summary>

Created: `2025-04-07T23:06:35Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/2nd62sq9>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "REx",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:06:20Z",
    "digest": "ed866c964387356285c586e9651b8fce",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjkzNzEwOA==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v8",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v8",
    "relationship": "logged",
    "size": 67443,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:06:22Z",
    "version": "v8"
  }
]
```

</details>

<details><summary>Run candidate 2: 8c74sww3 (chocolate-sweep-5)</summary>

Created: `2025-04-07T23:15:04Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/8c74sww3>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "REx",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:13:24Z",
    "digest": "45b83efc8a0483cb6d37454b1f7f4dfb",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjk0OTAxOQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v9",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v9",
    "relationship": "logged",
    "size": 67338,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:13:25Z",
    "version": "v9"
  }
]
```

</details>

## waterbirds_cf: GroupDRO

Paper target: in `0.906`, out `0.684`.

Strict schema filter: `2` finished runs; `2` expose a usable summary metric pair.
No metric-bearing candidate matched every logged appendix hyperparameter; ranking uses the strict schema-filtered fallback set.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `b8i9pme1` |    1 | 0.675819 | 0.172932 | -0.230181 | -0.511068 | 0.741249 |
|    2 | `pt634tp4` |    1 | 0.675819 | 0.172932 | -0.230181 | -0.511068 | 0.741249 |

**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.

<details><summary>Sweep/config aggregate 1: b8i9pme1</summary>

Metric source: `latest nested test summary`; run IDs: `on6mavlq`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "GroupDRO",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: pt634tp4</summary>

Metric source: `latest nested test summary`; run IDs: `ott0ewoh`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "GroupDRO",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name            | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | --------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `ott0ewoh` | glowing-sweep-6 | `pt634tp4` | latest nested test summary | 0.675819 / 0.172932 | -0.230181 / -0.511068 | 0.741249 |
|    2 | `on6mavlq` | autumn-sweep-6  | `b8i9pme1` | latest nested test summary | 0.675819 / 0.172932 | -0.230181 / -0.511068 | 0.741249 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: ott0ewoh (glowing-sweep-6)</summary>

Created: `2025-04-07T23:07:05Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/ott0ewoh>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "GroupDRO",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:06:20Z",
    "digest": "ed866c964387356285c586e9651b8fce",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjkzNzEwOA==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v8",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v8",
    "relationship": "logged",
    "size": 67443,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:06:22Z",
    "version": "v8"
  }
]
```

</details>

<details><summary>Run candidate 2: on6mavlq (autumn-sweep-6)</summary>

Created: `2025-04-07T23:15:35Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/on6mavlq>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "GroupDRO",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:13:24Z",
    "digest": "45b83efc8a0483cb6d37454b1f7f4dfb",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjk0OTAxOQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v9",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v9",
    "relationship": "logged",
    "size": 67338,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:13:25Z",
    "version": "v9"
  }
]
```

</details>

## waterbirds_cf: Fish

Paper target: in `0.900`, out `0.744`.

Strict schema filter: `2` finished runs; `2` expose a usable summary metric pair.
No metric-bearing candidate matched every logged appendix hyperparameter; ranking uses the strict schema-filtered fallback set.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `b8i9pme1` |    1 | 0.615330 | 0.191589 | -0.284670 | -0.552411 | 0.837082 |
|    2 | `pt634tp4` |    1 | 0.615330 | 0.191589 | -0.284670 | -0.552411 | 0.837082 |

**Near-tie:** more than one sweep/config aggregate is within 0.002 L1 distance of the best result; no unique sweep is asserted.

<details><summary>Sweep/config aggregate 1: b8i9pme1</summary>

Metric source: `latest nested test summary`; run IDs: `t88po4oh`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "Fish",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: pt634tp4</summary>

Metric source: `latest nested test summary`; run IDs: `riuytztc`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "solver": "Fish",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "Debug",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]
    },
    "epochs": {
      "values": [1]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "latent_dim": {
      "values": [256]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.1]
    },
    "param2": {
      "values": [10]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ERM", "ECMP", "IRM", "Fish", "REx", "GroupDRO"]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name             | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ---------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `riuytztc` | pleasant-sweep-4 | `pt634tp4` | latest nested test summary | 0.615330 / 0.191589 | -0.284670 / -0.552411 | 0.837082 |
|    2 | `t88po4oh` | copper-sweep-4   | `b8i9pme1` | latest nested test summary | 0.615330 / 0.191589 | -0.284670 / -0.552411 | 0.837082 |

**Near-tie:** at least two individual runs are within 0.002 L1 distance of the best run.

<details><summary>Run candidate 1: riuytztc (pleasant-sweep-4)</summary>

Created: `2025-04-07T23:06:04Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/riuytztc>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "Fish",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:06:20Z",
    "digest": "ed866c964387356285c586e9651b8fce",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjkzNzEwOA==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v8",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v8",
    "relationship": "logged",
    "size": 67443,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:06:22Z",
    "version": "v8"
  }
]
```

</details>

<details><summary>Run candidate 2: t88po4oh (copper-sweep-4)</summary>

Created: `2025-04-07T23:14:34Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/t88po4oh>.

Appendix-config mismatches: `['weight_decay=0 (paper: 0.0001)', 'epochs=1 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 1,
  "featurizer": "linear",
  "latent_dim": 256,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.1,
  "param2": 10,
  "param3": 0,
  "pretrained": "true",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "Fish",
  "split": "official",
  "upweighting": "false",
  "weight_decay": 0
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-07T23:13:24Z",
    "digest": "45b83efc8a0483cb6d37454b1f7f4dfb",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTYzNjk0OTAxOQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v9",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v9",
    "relationship": "logged",
    "size": 67338,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-07T23:13:25Z",
    "version": "v9"
  }
]
```

</details>

## waterbirds_cf: LISA

Paper target: in `0.904`, out `0.722`.

Strict schema filter: `4` finished runs; `4` expose a usable summary metric pair.
No metric-bearing candidate matched every logged appendix hyperparameter; ranking uses the strict schema-filtered fallback set.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `mqeob79m` |    1 | 0.893179 | 0.759399 | -0.010821 | +0.037399 | 0.048220 |
|    2 | `ejq3gd5r` |    1 | 0.889318 | 0.781955 | -0.014682 | +0.059955 | 0.074637 |
|    3 | `mqeob79m` |    1 | 0.889318 | 0.781955 | -0.014682 | +0.059955 | 0.074637 |

<details><summary>Sweep/config aggregate 1: mqeob79m</summary>

Metric source: `latest nested test summary`; run IDs: `xidklz5f`.

Shared run config:

```json
{
  "batch_size": 16,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.5,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-LISA-tuning",
  "parameters": {
    "batch_size": {
      "values": [16]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.3, 0.5, 0.7]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["LISA"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: ejq3gd5r</summary>

Metric source: `latest nested test summary`; run IDs: `ah24abww`.

Shared run config:

```json
{
  "batch_size": 16,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.3,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-LISA-tuning",
  "parameters": {
    "batch_size": {
      "values": [16]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.3, 0.5, 0.7]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["LISA"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: mqeob79m</summary>

Metric source: `latest nested test summary`; run IDs: `0h8vtu3p`.

Shared run config:

```json
{
  "batch_size": 16,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 0.3,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-LISA-tuning",
  "parameters": {
    "batch_size": {
      "values": [16]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [40]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [0.3, 0.5, 0.7]
    },
    "pretrained": {
      "values": ["true"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["LISA"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name               | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ------------------ | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `xidklz5f` | ancient-sweep-2    | `mqeob79m` | latest nested test summary | 0.893179 / 0.759399 | -0.010821 / +0.037399 | 0.048220 |
|    2 | `ah24abww` | dutiful-sweep-1    | `ejq3gd5r` | latest nested test summary | 0.889318 / 0.781955 | -0.014682 / +0.059955 | 0.074637 |
|    3 | `0h8vtu3p` | effortless-sweep-1 | `mqeob79m` | latest nested test summary | 0.889318 / 0.781955 | -0.014682 / +0.059955 | 0.074637 |

<details><summary>Run candidate 1: xidklz5f (ancient-sweep-2)</summary>

Created: `2025-04-23T16:58:46Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/xidklz5f>.

Appendix-config mismatches: `['batch_size=16 (paper: 256)', 'epochs=40 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 16,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.5,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

<details><summary>Run candidate 2: ah24abww (dutiful-sweep-1)</summary>

Created: `2025-04-23T16:22:25Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/ah24abww>.

Appendix-config mismatches: `['batch_size=16 (paper: 256)', 'epochs=40 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 16,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.3,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:22:44Z",
    "digest": "e8533d913c2945d8f756d0c17e6d0e4c",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzExOTI0Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v297",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v297",
    "relationship": "logged",
    "size": 161325,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:22:45Z",
    "version": "v297"
  }
]
```

</details>

<details><summary>Run candidate 3: 0h8vtu3p (effortless-sweep-1)</summary>

Created: `2025-04-23T16:55:00Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/0h8vtu3p>.

Appendix-config mismatches: `['batch_size=16 (paper: 256)', 'epochs=40 (paper: 100)']`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `0`.

Full run config:

```json
{
  "batch_size": 16,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 40,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 0.3,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "LISA",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[]
```

</details>

## waterbirds_cf: MatchDG (1NN, CNN)

Paper target: in `0.970`, out `0.080`.

Strict schema filter: `0` finished runs; `0` expose a usable summary metric pair.

No candidate can be selected from the exported summaries. This is reported as an unresolved row, not guessed.

## waterbirds_cf: MatchDG (1NN, Finetune)

Paper target: in `0.920`, out `0.112`.

Strict schema filter: `0` finished runs; `0` expose a usable summary metric pair.

No candidate can be selected from the exported summaries. This is reported as an unresolved row, not guessed.

## waterbirds_cf: MatchDG (random, Probing)

Paper target: in `0.793`, out `0.009`.

Strict schema filter: `4` finished runs; `4` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `8pogorao` |    1 | 0.772773 | 0.070093 | -0.020227 | +0.061093 | 0.081321 |
|    2 | `8pogorao` |    1 | 0.773631 | 0.076324 | -0.019369 | +0.067324 | 0.086693 |
|    3 | `8pogorao` |    1 | 0.775633 | 0.082555 | -0.017367 | +0.073555 | 0.090922 |

<details><summary>Sweep/config aggregate 1: 8pogorao</summary>

Metric source: `latest nested test summary`; run IDs: `ypiv11e6`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 14,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-MatchDG-conditional",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: 8pogorao</summary>

Metric source: `latest nested test summary`; run IDs: `3zr4o441`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 10,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-MatchDG-conditional",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: 8pogorao</summary>

Metric source: `latest nested test summary`; run IDs: `s4hrvm42`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 6,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-MatchDG-conditional",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name            | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | --------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `ypiv11e6` | exalted-sweep-4 | `8pogorao` | latest nested test summary | 0.772773 / 0.070093 | -0.020227 / +0.061093 | 0.081321 |
|    2 | `3zr4o441` | stoic-sweep-3   | `8pogorao` | latest nested test summary | 0.773631 / 0.076324 | -0.019369 / +0.067324 | 0.086693 |
|    3 | `s4hrvm42` | cosmic-sweep-2  | `8pogorao` | latest nested test summary | 0.775633 / 0.082555 | -0.017367 / +0.073555 | 0.090922 |

<details><summary>Run candidate 1: ypiv11e6 (exalted-sweep-4)</summary>

Created: `2025-04-23T17:30:52Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/ypiv11e6>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 14,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:21:40Z",
    "digest": "3166198bfea469671181470e58f07d44",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzIyODI1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v314",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v314",
    "relationship": "logged",
    "size": 164081,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:21:42Z",
    "version": "v314"
  }
]
```

</details>

<details><summary>Run candidate 2: 3zr4o441 (stoic-sweep-3)</summary>

Created: `2025-04-23T17:27:39Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/3zr4o441>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 10,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:21:40Z",
    "digest": "3166198bfea469671181470e58f07d44",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzIyODI1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v314",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v314",
    "relationship": "logged",
    "size": 164081,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:21:42Z",
    "version": "v314"
  }
]
```

</details>

<details><summary>Run candidate 3: s4hrvm42 (cosmic-sweep-2)</summary>

Created: `2025-04-23T17:24:37Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/s4hrvm42>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 6,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:21:40Z",
    "digest": "3166198bfea469671181470e58f07d44",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzIyODI1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v314",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v314",
    "relationship": "logged",
    "size": 164081,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:21:42Z",
    "version": "v314"
  }
]
```

</details>

## waterbirds_cf: MatchDG (1NN, Probing)

Paper target: in `0.886`, out `0.411`.

Strict schema filter: `5` finished runs; `4` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `ld910a42` |    1 | 0.885028 | 0.400312 | -0.000972 | -0.010688 | 0.011661 |
|    2 | `ld910a42` |    1 | 0.846704 | 0.130841 | -0.039296 | -0.280159 | 0.319455 |
|    3 | `ld910a42` |    1 | 0.778350 | 0.000000 | -0.107650 | -0.411000 | 0.518650 |

<details><summary>Sweep/config aggregate 1: ld910a42</summary>

Metric source: `latest nested test summary`; run IDs: `ro69rgv2`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-MatchDG-nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: ld910a42</summary>

Metric source: `latest nested test summary`; run IDs: `b5vjd2ua`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 6,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-MatchDG-nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: ld910a42</summary>

Metric source: `latest nested test summary`; run IDs: `ypuxp5sw`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 10,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-MatchDG-nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name            | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | --------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `ro69rgv2` | rural-sweep-1   | `ld910a42` | latest nested test summary | 0.885028 / 0.400312 | -0.000972 / -0.010688 | 0.011661 |
|    2 | `b5vjd2ua` | hopeful-sweep-2 | `ld910a42` | latest nested test summary | 0.846704 / 0.130841 | -0.039296 / -0.280159 | 0.319455 |
|    3 | `ypuxp5sw` | decent-sweep-3  | `ld910a42` | latest nested test summary | 0.778350 / 0.000000 | -0.107650 / -0.411000 | 0.518650 |

<details><summary>Run candidate 1: ro69rgv2 (rural-sweep-1)</summary>

Created: `2025-04-23T17:22:14Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/ro69rgv2>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:21:40Z",
    "digest": "3166198bfea469671181470e58f07d44",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzIyODI1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v314",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v314",
    "relationship": "logged",
    "size": 164081,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:21:42Z",
    "version": "v314"
  }
]
```

</details>

<details><summary>Run candidate 2: b5vjd2ua (hopeful-sweep-2)</summary>

Created: `2025-04-23T17:25:00Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/b5vjd2ua>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 6,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:21:40Z",
    "digest": "3166198bfea469671181470e58f07d44",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzIyODI1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v314",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v314",
    "relationship": "logged",
    "size": 164081,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:21:42Z",
    "version": "v314"
  }
]
```

</details>

<details><summary>Run candidate 3: ypuxp5sw (decent-sweep-3)</summary>

Created: `2025-04-23T17:27:52Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/ypuxp5sw>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 10,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:21:40Z",
    "digest": "3166198bfea469671181470e58f07d44",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzIyODI1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v314",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v314",
    "relationship": "logged",
    "size": 164081,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:21:42Z",
    "version": "v314"
  }
]
```

</details>

## waterbirds_cf: MatchDG (clean, Probing)

Paper target: in `0.906`, out `0.536`.

Strict schema filter: `4` finished runs; `4` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `78mawi7i` |    1 | 0.893465 | 0.637072 | -0.012535 | +0.101072 | 0.113607 |
|    2 | `78mawi7i` |    1 | 0.897183 | 0.651090 | -0.008817 | +0.115090 | 0.123907 |
|    3 | `78mawi7i` |    1 | 0.897040 | 0.651090 | -0.008960 | +0.115090 | 0.124050 |

<details><summary>Sweep/config aggregate 1: 78mawi7i</summary>

Metric source: `latest nested test summary`; run IDs: `y5q1vhgq`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-MatchDG-oracle",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: 78mawi7i</summary>

Metric source: `latest nested test summary`; run IDs: `aj7jh8pv`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 14,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-MatchDG-oracle",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: 78mawi7i</summary>

Metric source: `latest nested test summary`; run IDs: `qie6044d`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 10,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-MatchDG-oracle",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["MatchDG"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name             | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ---------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `y5q1vhgq` | usual-sweep-1    | `78mawi7i` | latest nested test summary | 0.893465 / 0.637072 | -0.012535 / +0.101072 | 0.113607 |
|    2 | `aj7jh8pv` | fragrant-sweep-4 | `78mawi7i` | latest nested test summary | 0.897183 / 0.651090 | -0.008817 / +0.115090 | 0.123907 |
|    3 | `qie6044d` | spring-sweep-3   | `78mawi7i` | latest nested test summary | 0.897040 / 0.651090 | -0.008960 / +0.115090 | 0.124050 |

<details><summary>Run candidate 1: y5q1vhgq (usual-sweep-1)</summary>

Created: `2025-04-23T17:21:25Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/y5q1vhgq>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:21:40Z",
    "digest": "3166198bfea469671181470e58f07d44",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzIyODI1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v314",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v314",
    "relationship": "logged",
    "size": 164081,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:21:42Z",
    "version": "v314"
  }
]
```

</details>

<details><summary>Run candidate 2: aj7jh8pv (fragrant-sweep-4)</summary>

Created: `2025-04-23T17:31:36Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/aj7jh8pv>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 14,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:21:40Z",
    "digest": "3166198bfea469671181470e58f07d44",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzIyODI1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v314",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v314",
    "relationship": "logged",
    "size": 164081,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:21:42Z",
    "version": "v314"
  }
]
```

</details>

<details><summary>Run candidate 3: qie6044d (spring-sweep-3)</summary>

Created: `2025-04-23T17:27:43Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/qie6044d>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 10,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "MatchDG",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:21:40Z",
    "digest": "3166198bfea469671181470e58f07d44",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzIyODI1Mw==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v314",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v314",
    "relationship": "logged",
    "size": 164081,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:21:42Z",
    "version": "v314"
  }
]
```

</details>

## waterbirds_cf: GRIT (random, Probing)

Paper target: in `0.804`, out `0.269`.

Strict schema filter: `4` finished runs; `4` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `ep2y89o3` |    1 | 0.803947 | 0.269470 | -0.000053 | +0.000470 | 0.000524 |
|    2 | `ep2y89o3` |    1 | 0.792364 | 0.138629 | -0.011636 | -0.130371 | 0.142007 |
|    3 | `ep2y89o3` |    1 | 0.730159 | 0.191589 | -0.073841 | -0.077411 | 0.151252 |

<details><summary>Sweep/config aggregate 1: ep2y89o3</summary>

Metric source: `latest nested test summary`; run IDs: `i34a6ite`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 6,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-ECMP-conditional",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: ep2y89o3</summary>

Metric source: `latest nested test summary`; run IDs: `8ntovjdy`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 10,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-ECMP-conditional",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: ep2y89o3</summary>

Metric source: `latest nested test summary`; run IDs: `372jvuun`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-ECMP-conditional",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [2, 6, 10, 14]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["conditional"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name             | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ---------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `i34a6ite` | bumbling-sweep-2 | `ep2y89o3` | latest nested test summary | 0.803947 / 0.269470 | -0.000053 / +0.000470 | 0.000524 |
|    2 | `8ntovjdy` | wise-sweep-3     | `ep2y89o3` | latest nested test summary | 0.792364 / 0.138629 | -0.011636 / -0.130371 | 0.142007 |
|    3 | `372jvuun` | fast-sweep-1     | `ep2y89o3` | latest nested test summary | 0.730159 / 0.191589 | -0.073841 / -0.077411 | 0.151252 |

<details><summary>Run candidate 1: i34a6ite (bumbling-sweep-2)</summary>

Created: `2025-04-23T17:03:20Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/i34a6ite>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 6,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:03:31Z",
    "digest": "91e5d81ca59cebc0baefca9690531f00",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzE5NjUxMQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v313",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v313",
    "relationship": "logged",
    "size": 162376,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:03:33Z",
    "version": "v313"
  }
]
```

</details>

<details><summary>Run candidate 2: 8ntovjdy (wise-sweep-3)</summary>

Created: `2025-04-23T17:06:18Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/8ntovjdy>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 10,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:03:31Z",
    "digest": "91e5d81ca59cebc0baefca9690531f00",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzE5NjUxMQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v313",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v313",
    "relationship": "logged",
    "size": 162376,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:03:33Z",
    "version": "v313"
  }
]
```

</details>

<details><summary>Run candidate 3: 372jvuun (fast-sweep-1)</summary>

Created: `2025-04-23T17:00:23Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/372jvuun>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "conditional",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:00:22Z",
    "digest": "c9c53e7341d005d05fd7592bde5c45a8",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzE5MDkyNg==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v312",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v312",
    "relationship": "logged",
    "size": 162313,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:00:24Z",
    "version": "v312"
  }
]
```

</details>

## waterbirds_cf: GRIT (1NN, Probing)

Paper target: in `0.892`, out `0.521`.

Strict schema filter: `5` finished runs; `5` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `jrd0pl4w` |    1 | 0.892178 | 0.521807 | +0.000178 | +0.000807 | 0.000985 |
|    2 | `jrd0pl4w` |    1 | 0.891463 | 0.510903 | -0.000537 | -0.010097 | 0.010634 |
|    3 | `jrd0pl4w` |    1 | 0.876591 | 0.531153 | -0.015409 | +0.010153 | 0.025562 |

<details><summary>Sweep/config aggregate 1: jrd0pl4w</summary>

Metric source: `latest nested test summary`; run IDs: `pc0ihs74`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 20,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-ECMP-nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [4, 8, 12, 16, 20]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: jrd0pl4w</summary>

Metric source: `latest nested test summary`; run IDs: `roqj2wj9`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 16,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-ECMP-nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [4, 8, 12, 16, 20]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: jrd0pl4w</summary>

Metric source: `latest nested test summary`; run IDs: `sfnwep2n`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 4,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-ECMP-nearest",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [4, 8, 12, 16, 20]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["nearest"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name            | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | --------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `pc0ihs74` | swept-sweep-5   | `jrd0pl4w` | latest nested test summary | 0.892178 / 0.521807 | +0.000178 / +0.000807 | 0.000985 |
|    2 | `roqj2wj9` | exalted-sweep-4 | `jrd0pl4w` | latest nested test summary | 0.891463 / 0.510903 | -0.000537 / -0.010097 | 0.010634 |
|    3 | `sfnwep2n` | vocal-sweep-1   | `jrd0pl4w` | latest nested test summary | 0.876591 / 0.531153 | -0.015409 / +0.010153 | 0.025562 |

<details><summary>Run candidate 1: pc0ihs74 (swept-sweep-5)</summary>

Created: `2025-04-23T17:03:14Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/pc0ihs74>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 20,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:03:31Z",
    "digest": "91e5d81ca59cebc0baefca9690531f00",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzE5NjUxMQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v313",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v313",
    "relationship": "logged",
    "size": 162376,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:03:33Z",
    "version": "v313"
  }
]
```

</details>

<details><summary>Run candidate 2: roqj2wj9 (exalted-sweep-4)</summary>

Created: `2025-04-23T17:00:07Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/roqj2wj9>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 16,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T17:00:22Z",
    "digest": "c9c53e7341d005d05fd7592bde5c45a8",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzE5MDkyNg==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v312",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v312",
    "relationship": "logged",
    "size": 162313,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T17:00:24Z",
    "version": "v312"
  }
]
```

</details>

<details><summary>Run candidate 3: sfnwep2n (vocal-sweep-1)</summary>

Created: `2025-04-23T16:50:26Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/sfnwep2n>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 4,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "nearest",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-04-23T16:48:39Z",
    "digest": "e569fa6c183ddbfaa3f38a17960d3fb8",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTY3NzE2OTM4Ng==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v310",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v310",
    "relationship": "logged",
    "size": 161630,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-04-23T16:48:41Z",
    "version": "v310"
  }
]
```

</details>

## waterbirds_cf: GRIT (clean, Probing)

Paper target: in `0.864`, out `0.812`.

Strict schema filter: `7` finished runs; `7` expose a usable summary metric pair.
Restricted to candidates matching all logged appendix hyperparameters.

### Candidate sweep/config aggregates

Aggregates group runs by sweep, non-volatile config, and metric source; seeds are averaged only within an identical config.

| Rank | Sweep      | Runs |  In mean | Out mean |      Δ in |     Δ out |       L1 |
| ---: | ---------- | ---: | -------: | -------: | --------: | --------: | -------: |
|    1 | `ndihtoy9` |    1 | 0.911626 | 0.643302 | +0.047626 | -0.168698 | 0.216324 |
|    2 | `ndihtoy9` |    1 | 0.910625 | 0.637072 | +0.046625 | -0.174928 | 0.221553 |
|    3 | `ndihtoy9` |    1 | 0.902045 | 0.627726 | +0.038045 | -0.184274 | 0.222319 |

<details><summary>Sweep/config aggregate 1: ndihtoy9</summary>

Metric source: `latest nested test summary`; run IDs: `6e7x6weq`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 12,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-ECMP-oracle_r",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [1, 2, 4, 8, 12, 16, 20]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 2: ndihtoy9</summary>

Metric source: `latest nested test summary`; run IDs: `oux6x3j3`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 20,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-ECMP-oracle_r",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [1, 2, 4, 8, 12, 16, 20]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

<details><summary>Sweep/config aggregate 3: ndihtoy9</summary>

Metric source: `latest nested test summary`; run IDs: `mcy0c7og`.

Shared run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Full sweep config:

```json
{
  "method": "grid",
  "metric": {
    "goal": "maximize",
    "name": "test.acc_avg"
  },
  "name": "CounterfactualWaterbirds-ECMP-oracle_r",
  "parameters": {
    "batch_size": {
      "values": [256]
    },
    "dataset": {
      "values": ["CounterfactualWaterbirds"]
    },
    "epochs": {
      "values": [100]
    },
    "featurizer": {
      "values": ["linear"]
    },
    "lr": {
      "values": [0.001]
    },
    "param1": {
      "values": [1, 2, 4, 8, 12, 16, 20]
    },
    "pretrained": {
      "values": ["true"]
    },
    "projection": {
      "values": ["oracle"]
    },
    "seed": {
      "values": [1001]
    },
    "solver": {
      "values": ["ECMP"]
    },
    "weight_decay": {
      "values": [0.0001]
    }
  },
  "program": "main.py"
}
```

</details>

### Top individual runs

| Rank | Run        | Name             | Sweep      | Metric source              | Logged in/out       | Δ in/out              |       L1 |
| ---: | ---------- | ---------------- | ---------- | -------------------------- | ------------------- | --------------------- | -------: |
|    1 | `6e7x6weq` | misty-sweep-5    | `ndihtoy9` | latest nested test summary | 0.911626 / 0.643302 | +0.047626 / -0.168698 | 0.216324 |
|    2 | `oux6x3j3` | bright-sweep-7   | `ndihtoy9` | latest nested test summary | 0.910625 / 0.637072 | +0.046625 / -0.174928 | 0.221553 |
|    3 | `mcy0c7og` | fanciful-sweep-2 | `ndihtoy9` | latest nested test summary | 0.902045 / 0.627726 | +0.038045 / -0.184274 | 0.222319 |

<details><summary>Run candidate 1: 6e7x6weq (misty-sweep-5)</summary>

Created: `2025-07-24T14:50:06Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/6e7x6weq>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 12,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-07-24T14:45:37Z",
    "digest": "a2faf2e1a5bf8734294227d04549acdf",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTkwMjUwMDI3OQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v338",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v338",
    "relationship": "logged",
    "size": 177936,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-07-24T14:45:39Z",
    "version": "v338"
  }
]
```

</details>

<details><summary>Run candidate 2: oux6x3j3 (bright-sweep-7)</summary>

Created: `2025-07-24T14:52:12Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/oux6x3j3>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 20,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-07-24T14:45:37Z",
    "digest": "a2faf2e1a5bf8734294227d04549acdf",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTkwMjUwMDI3OQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v338",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v338",
    "relationship": "logged",
    "size": 177936,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-07-24T14:45:39Z",
    "version": "v338"
  }
]
```

</details>

<details><summary>Run candidate 3: mcy0c7og (fanciful-sweep-2)</summary>

Created: `2025-07-24T14:47:49Z`; state: `finished`; URL: <https://wandb.ai/inouye-lab/CMP/runs/mcy0c7og>.

Appendix-config mismatches: `none`.

Provenance warnings: `['no W&B dataset artifact/hash is attached', 'optimizer is absent from W&B config and cannot be verified from config alone']`.

Logged dataset artifacts: `0`; code artifacts: `1`.

Full run config:

```json
{
  "batch_size": 256,
  "dataset": "CounterfactualWaterbirds",
  "epochs": 100,
  "featurizer": "linear",
  "latent_dim": 512,
  "lr": 0.001,
  "no_wandb": false,
  "param1": 2,
  "param2": 0,
  "param3": 0,
  "pretrained": "true",
  "projection": "oracle",
  "root_dir": "/local/scratch/a/bai116/datasets/",
  "seed": 1001,
  "solver": "ECMP",
  "split_scheme": "official",
  "upweighting": "false",
  "weight_decay": 0.0001
}
```

Dataset artifacts:

```json
[]
```

Code artifacts:

```json
[
  {
    "aliases": [],
    "created_at": "2025-07-24T14:45:37Z",
    "digest": "a2faf2e1a5bf8734294227d04549acdf",
    "entity": "inouye-lab",
    "id": "QXJ0aWZhY3Q6MTkwMjUwMDI3OQ==",
    "is_dataset": false,
    "name": "source-CMP-main.py:v338",
    "project": "CMP",
    "qualified_name": "inouye-lab/CMP/source-CMP-main.py:v338",
    "relationship": "logged",
    "size": 177936,
    "state": "COMMITTED",
    "type": "code",
    "updated_at": "2025-07-24T14:45:39Z",
    "version": "v338"
  }
]
```

</details>
