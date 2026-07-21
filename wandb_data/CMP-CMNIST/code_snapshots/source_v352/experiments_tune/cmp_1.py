import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "grid",
    "name": "CMP_1_tuning",
    "metric": {
        "goal": "maximize",
        "name": "test_acc"
        },
    "parameters": {
        "solver": {"values":["CMP"]},
        "mode": {"values":[1]},
        "latent_dim": {"values":[16,32,64,128]},
        "param1": {"values":[1,10,100,1000]},
        "param2": {"values":[10000]},
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
