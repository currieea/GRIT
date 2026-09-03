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
        "solver": {"values":["ECMP"]},
        "lr": {'values':[0.001]},
        "param1": {'values': [0, 2, 4, 6, 8, 10]},
        "batch_size": {"values":[256]},
        "latent_dim": {"values":[256]},
        "epochs": {"values":[300]},
        "seed": {"values":[1001]},
        "featurizer": {"values": ["linear"]},
        "pretrained": {"values": ["true"]},
        "dataset": {"values": ["CounterfactualWaterbirds", "CounterfactualCelebA"]},
     },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
