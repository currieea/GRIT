import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CounterfactualWaterbirds-ECMP-oracle",
    "metric": {
        "goal": "maximize",
        "name": "test.acc_avg"
        },
    "parameters": {
        "solver": {"values":["ECMP"]},
        "lr": {'values':[0.001]},
        "param1": {'values': [4,8,12,16,20]},
        "weight_decay": {'values': [1e-4]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[100]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "projection": {"values": ["oracle"]},
        "dataset": {"values": ["CounterfactualWaterbirds"]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
