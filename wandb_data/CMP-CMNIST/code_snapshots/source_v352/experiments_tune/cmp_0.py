import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CMP_0_MC",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["CMP"]},
        "param1": {"values":[100000]},
        "param2": {"values":[1000]},
        "mode": {"values":[0]},
        "lr": {"values":[2e-4]},
        "batch_size": {"values":[256]},
        "fewshot_batch_size": {"values":[256]},
        "epochs": {"values":[4]},
        "seed": {"values":[1001, 1002, 1003, 1004, 1005, 1006,1007, 1008, 1009, 1010]},
        "latent_dim": {"values":[64]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP-CMNIST", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
