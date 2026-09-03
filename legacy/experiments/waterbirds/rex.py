import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CounterfactualWaterbirds-REx-tuning",
    "metric": {
        "goal": "maximize",
        "name": "test.acc_avg"
        },
    "parameters": {
        "solver": {"values":["REx"]},
        "lr": {'values':[0.001]},
        "param1": {'values': [1,10,100,1000,10000]},
        "param2": {'values': [100]},
        "weight_decay": {'values': [1e-4]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "projection": {"values": ["conditional"]},
        "dataset": {"values": ["CounterfactualWaterbirds"]},
        "seed": {"values": [1001]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
