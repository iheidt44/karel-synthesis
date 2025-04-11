import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from dsl import DSL
from logger.stdout_logger import StdoutLogger
from vae.models import load_model
from search.latent_search import LatentSearch
from tasks import get_task_cls
import pandas as pd
import os
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)

Config.search_number_iterations = 40
Config.search_number_executions = 6
Config.search_sigma = 0.25
Config.disable_gpu = True
Config.env_enable_leaps_behaviour = True
Config.model_hidden_size = 256
# Config.model_params_path = "output/semantic_only_256_v2/model/best_val.ptp"
# Config.env_task = "Maze"
Config.model_name = "LeapsVAE"
Config.search_reduce_to_mean = False
Config.search_population_size = 256
Config.search_elitism_rate = 0.0625

tasks = {
    "KAREL": ["StairClimber", "Maze", "TopOff", "FourCorners", "Harvester"],
    "KAREL-HARD": ["DoorKey", "OneStroke", "Seeder", "Snake"]
}

# tasks = {
#     "KAREL": ["StairClimber", "Maze", "TopOff", "FourCorners", "Harvester"]
# }

# tasks = {
#     "KAREL": ["DoorKey"]
# }

all_tasks = tasks["KAREL"] + tasks["KAREL-HARD"]
num_seeds = 3
methods = {
    "LEAPS": {"params_path": "params/leaps_vae_256.ptp", "use_latent_only": False},
    "NewLatent": {"params_path": "output/semantic_only_256_v2/model/best_val.ptp", "use_latent_only": True}
}

results = {task: {method: {"final_returns": [], "eval_counts": [], "curves": [], "num_restarts":[]} for method in methods} for task in all_tasks}


dsl = DSL.init_default_karel()
device = torch.device('cpu') if Config.disable_gpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for task in all_tasks:
    Config.env_task = task
    task_cls = get_task_cls(task)
    
    for method, config in methods.items():
        Config.model_params_path = config["params_path"]
        Config.model_name = "LeapsVAE"
        use_latent_only = config["use_latent_only"]
        
        for seed in range(num_seeds):
            Config.env_seed = seed
            Config.model_seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = load_model(Config.model_name, dsl, device)
            params = torch.load(Config.model_params_path, map_location=device)
            model.load_state_dict(params, strict=False)
            model.eval()
            
            searcher = LatentSearch(model, task_cls, dsl)
            StdoutLogger.log('Main', f'Task: {task}, Method: {method}, Seed: {seed+1}/{num_seeds}')
            
            best_program, converged, num_evals, reward_curve, num_restarts = searcher.search(use_latent_only=use_latent_only, return_curve=True)
            
            results[task][method]["final_returns"].append(searcher.best_reward)
            results[task][method]["eval_counts"].append(num_evals)
            results[task][method]["curves"].append(reward_curve)
            results[task][method]["num_restarts"].append(num_restarts)
            StdoutLogger.log('Main', f'Task: {task}, Method: {method}, Seed: {seed+1}, Reward: {searcher.best_reward}, Evals: {num_evals}, Num Restarts: {num_restarts}')

    data = []
    for method in methods:
        for seed in range(num_seeds):
            data.append({
                "Method": method,
                "Seed": seed + 1,
                "Final_Return": results[task][method]["final_returns"][seed],
                "Evaluations": results[task][method]["eval_counts"][seed],
                "Reward_Curve": results[task][method]["curves"][seed],
                "Num_Restarts": results[task][method]["num_restarts"][seed]
            })
    df = pd.DataFrame(data)
    os.makedirs("experiment_data", exist_ok=True)
    df.to_csv(f"experiment_data/{task}_results1.csv", index=False)

table_data = []
for task in all_tasks:
    row = {"Task": task}
    for method in methods:
        returns = np.array(results[task][method]["final_returns"])
        mean = returns.mean()
        std_err = returns.std(ddof=1) / np.sqrt(num_seeds)
        row[f"{method}"] = f"{mean:.2f} ± {std_err:.2f}"
    table_data.append(row)
table_df = pd.DataFrame(table_data)
print("\nTable 1: Mean and Standard Error of Final Episodic Returns")
print(table_df.to_string(index=False))
table_df.to_csv("table1_results.csv", index=False)

plt.figure(figsize=(15, 10))
for i, task in enumerate(all_tasks):
    plt.subplot(4, 3, i + 1)
    print(f"\nProcessing task: {task}")
    
    for method in methods:
        curves = results[task][method]["curves"]
        if not curves:
            print(f"No curves for {method} in {task}")
            continue
        
        max_evals = max(max(curve[-1][1] for curve in curves if curve), 1)
        x_eval = np.logspace(0, np.log10(max_evals), 100)
        print(f"{method}: max_evals={max_evals}, x_eval range={x_eval[0]} to {x_eval[-1]}")
        
        interp_curves = []
        for curve in curves:
            if not curve:
                continue
            rewards, evals = zip(*curve)
            interp = np.interp(x_eval, evals, rewards, left=rewards[0], right=rewards[-1])
            interp_curves.append(interp)
            print(f"INTERP {method}: {interp[:5]}... to {interp[-5:]}")
        
        if not interp_curves:
            continue
        interp_curves = np.array(interp_curves)
        mean_curve = interp_curves.mean(axis=0)
        se_curve = interp_curves.std(axis=0, ddof=1) / np.sqrt(num_seeds)
        print(f"{method} mean: {mean_curve[:5]}... to {mean_curve[-5:]}")
        plt.plot(x_eval, mean_curve, label=method, linewidth=2)
        plt.fill_between(x_eval, mean_curve - 1.96 * se_curve, mean_curve + 1.96 * se_curve, alpha=0.2)
    
    plt.xscale('log')
    plt.xlabel("Evaluations")
    plt.ylabel("Best Return")
    plt.title(task)
    plt.legend()
   
    all_rewards = [r for method in methods for curve in results[task][method]["curves"] for r, _ in curve if curve]
    if all_rewards:
        plt.ylim(min(-0.5, min(all_rewards)), max(1.1, max(all_rewards)))
        print(f"Y-limits for {task}: {min(-0.5, min(all_rewards))} to {max(1.1, max(all_rewards))}")
    else:
        plt.ylim(-0.5, 1.1)

plt.tight_layout()
plt.savefig("figure3_convergence.png")
plt.show()
