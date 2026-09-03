import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "ColoredMNIST-MatchDG-Nearest",
    "metric": {
        "goal": "maximize",
        "name": "test.acc_avg"
        },
    "parameters": {
        "solver": {"values":["MatchDG"]},
        "lr": {'values':[0.001]},
        "param1": {'values': [0.01, 0.1 ,1,10,100]},
        "weight_decay": {'values': [1e-4]},
        "batch_size": {"values":[256]},
        "epochs": {"values":[40]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "projection": {"values": ["nearest"]},
        "latent_dim": {"values": [4,8,12,16,20,24,28,32]},
        "dataset": {"values": ["LISAColoredMNIST"]},
        "seed": {"values": [1001]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
