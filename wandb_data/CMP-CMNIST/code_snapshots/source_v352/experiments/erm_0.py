import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "bayes",
    "name": "ERM_0_tuning",
    "metric": {
        "goal": "maximize",
        "name": "in_test_acc"
        },
    "parameters": {
        "solver": {"values":["ERM"]},
        "lr": {'distribution': 'log_uniform_values',
               'min': 1e-5,
               'max': 1e-3},
        "batch_size": {"values":[256]},
        "latent_dim": {'values':[128]},
        "epochs": {"values":[4]},
        "seed": {"values":[1001]},
        "mode": {"values": [0]}
     },
    "run_cap": 10
}


sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP-CMNIST", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)