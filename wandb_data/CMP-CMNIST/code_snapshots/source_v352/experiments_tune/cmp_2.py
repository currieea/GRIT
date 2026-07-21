import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CMP_2_MC",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "mode": {"values":[2]},
        "solver": {"values":["CMP"]},
        "latent_dim": {"values":[16]},
        "param1": {"values":[50]},
        "param2": {"values":[100,300,500,700,1000,3000]},
        "lr": {"values":[1e-4]},
        "batch_size": {"values":[256]},
        "fewshot_batch_size": {"values":[128]},
        "epochs": {"values":[50]},
        "seed": {"values":[1001]},
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP-CMNIST", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
