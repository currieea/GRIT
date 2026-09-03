import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "RotatedMNIST-LISA-tuning",
    "metric": {
        "goal": "maximize",
        "name": "test.acc_avg"
        },
    "parameters": {
        "solver": {"values":["LISA"]},
        "lr": {'values':[0.001]},
        "param1": {'values': [0,0.1,0.3,0.5,0.7,0.9,1]},
        "weight_decay": {'values': [1e-4]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "dataset": {"values": ["RotatedMNIST"]},
        "seed": {"values": [1001]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
