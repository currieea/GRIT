import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "ColoredMNIST-GroupDRO-sweep",
    "metric": {
        "goal": "maximize",
        "name": "test.acc_avg"
        },
    "parameters": {
        "solver": {"values":["GroupDRO"]},
        "lr": {'values':[0.001]},
        "param1": {'values': [0.001,0.1]},
        "weight_decay": {'values': [1e-4]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "projection": {"values": ["conditional"]},
        "dataset": {"values": ["ColoredMNIST"]},
        "seed": {"values": [1001,1002,1003,1004,1005,1006, 1007,1008,1009,1010]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
