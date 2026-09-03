import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "ColoredMNIST-SWAD-tuning",
    "metric": {
        "goal": "maximize",
        "name": "test.acc_avg"
        },
    "parameters": {
        "solver": {"values":["SWAD"]},
        "lr": {'values':[0.001]},
        "param1": {'values': [1.1,1.2,1.3,1.4,1.5]},
        "weight_decay": {'values': [1e-4]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "dataset": {"values": ["LISAColoredMNIST"]},
        "seed": {"values": [1001]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
