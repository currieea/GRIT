import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "ColoredMNIST-ERM",
    "metric": {
        "goal": "maximize",
        "name": "test.acc_avg"
        },
    "parameters": {
        "solver": {"values":["ERM"]},
        "lr": {'values':[0.001]},
        # "param1": {'values': [0,1,2,3,4,5,6,7,8,9]},
        # "param2": {'values': [512]},
        "weight_decay": {'values': [1e-4]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "projection": {"values": ["oracle"]},
        "dataset": {"values": ["ColoredMNIST"]},
        "seed": {"values": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
