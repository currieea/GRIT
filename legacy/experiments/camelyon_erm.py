import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "Debug",
    "metric": {
        "goal": "maximize",
        "name": "test.acc_avg"
        },
    "parameters": {
        "solver": {"values":["ERM"]},
        "lr": {'values':[0.001]},
        "param1": {'values': [0.001,0.01,0.1]},
        "weight_decay": {'values': [0, 1e-4,1e-3,1e-2]},
        "batch_size": {"values":[256]},
        "latent_dim": {"values":[128]},
        "epochs": {"values":[100]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "dataset": {"values": ["Camelyon17"]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
