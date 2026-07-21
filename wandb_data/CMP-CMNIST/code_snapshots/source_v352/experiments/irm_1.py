import wandb


sweep_configuration = {
    "program": "main.py",
    "method": "bayes",
    "name": "IRM_1_tuning",
    "metric": {
        "goal": "maximize",
        "name": "in_test_acc"
        },
    "parameters": {
        "solver": {"values":["IRM"]},
        "lr": {'distribution': 'log_uniform_values',
               'min': 1e-5,
               'max': 1e-3},
        "batch_size": {"values":[256]},
        "latent_dim": {'values':[128]},
        "epochs": {"values":[4]},
        "seed": {"values":[1001]},
        "param1": {'distribution': 'log_uniform_values',
               'min': 0.1,
               'max': 100000},
        "param2": {'distribution': 'int_uniform',
               'min': 1,
               'max': 10000},
        "param3": {'distribution': 'log_uniform_values',
               'min': 1e-4,
               'max': 1e-2},
        "mode": {"values": [1]}
     },
    "early_terminate": False,  
    "run_cap": 160
}



sweep_id = wandb.sweep(sweep=sweep_configuration, project="CMP-CMNIST", entity="inouye-lab")
print(sweep_id)
wandb.agent(sweep_id)