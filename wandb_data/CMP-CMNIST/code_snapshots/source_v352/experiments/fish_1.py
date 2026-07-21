import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "bayes",
    "name": "Fish_1_tuning",
    "metric": {
        "goal": "maximize",
        "name": "in_test_acc"
        },
    "parameters": {
        "solver": {"values":["Fish"]},
        "lr": {'distribution': 'log_uniform_values',
               'min': 1e-5,
               'max': 1e-3},
        "batch_size": {"values":[256]},
        "latent_dim": {'values':[128]},
        "epochs": {"values":[4]},
        "seed": {"values":[1001]},
        "param1": {'distribution': 'log_uniform_values',
               'min': 0.0001,
               'max': 0.1},
        "mode": {"values": [1]}
     },
    "run_cap": 160
}



sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP-CMNIST", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)