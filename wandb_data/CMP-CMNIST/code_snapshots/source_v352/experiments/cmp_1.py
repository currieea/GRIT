import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "bayes",
    "name": "CMP_1_tuning",
    "metric": {
        "goal": "maximize",
        "name": "in_test_acc"
        },
    "parameters": {
        "solver": {"values":["CMP"]},
        "lr": {'distribution': 'log_uniform_values',
               'min': 1e-5,
               'max': 1e-3},
        "param1": {'distribution': 'log_uniform_values',
               'min': 0.1,
               'max': 100000},
        "param2": {'values':[1000]},
        "batch_size": {"values":[256]},
        "latent_dim": {'distribution': 'q_log_uniform_values',
               'min': 2,
               'max': 128},
        "epochs": {"values":[10]},
        "seed": {"values":[1001]},
        "mode": {"values": [1]}
     },
    "run_cap": 36
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP-CMNIST", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)
