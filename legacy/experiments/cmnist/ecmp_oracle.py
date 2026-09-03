import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "ColoredMNIST-ECMP-Oracle",
    "metric": {
        "goal": "maximize",
        "name": "test.acc_avg"
        },
    "parameters": {
        "solver": {"values":["ECMP"]},
        "lr": {'values':[0.001]},
        "param1": {'values': [2,4,6,8,10,12]},
        "param2": {'values': [1024]},
        "weight_decay": {'values': [1e-4]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "projection": {"values": ["oracle"]},
        "dataset": {"values": ["LISAColoredMNIST"]},
        "seed": {"values": [1001]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
