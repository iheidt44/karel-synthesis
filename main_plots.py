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

# Parameters for evaluation
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

# Table 1
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

# Figure 3
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
        
        # Interpolated mean curve (no individual curves plotted)
        interp_curves = []
        for curve in curves:
            if not curve:
                continue
            rewards, evals = zip(*curve)  # rewards first, evals second
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
    # plt.grid(True, which="both")
    # Set y-axis limits to capture full reward range
    all_rewards = [r for method in methods for curve in results[task][method]["curves"] for r, _ in curve if curve]
    if all_rewards:
        plt.ylim(min(-0.5, min(all_rewards)), max(1.1, max(all_rewards)))
        print(f"Y-limits for {task}: {min(-0.5, min(all_rewards))} to {max(1.1, max(all_rewards))}")
    else:
        plt.ylim(-0.5, 1.1)

plt.tight_layout()
plt.savefig("figure3_convergence.png")
plt.show()

# # Figure 3
# plt.figure(figsize=(15, 10))
# for i, task in enumerate(all_tasks):
#     plt.subplot(4, 3, i + 1)
#     print(f"\nProcessing task: {task}")
    
#     for method in methods:
#         curves = results[task][method]["curves"]
#         if not curves:
#             print(f"No curves for {method} in {task}")
#             continue
        
#         max_evals = max(max(curve[-1][1] for curve in curves if curve), 1)
#         x_eval = np.logspace(0, np.log10(max_evals), 100)
#         print(f"{method}: max_evals={max_evals}, x_eval range={x_eval[0]} to {x_eval[-1]}")
        
#         # Plot raw step curves for each seed (faint)
#         for j, curve in enumerate(curves):
#             if not curve:
#                 print(f"Empty curve for {method}, seed {j}")
#                 continue
#             rewards, evals = zip(*curve)
#             print(f"CURVE {method} Seed {j+1}: evals={evals[:5]}..., rewards={rewards[:5]}...")
#             plt.step(evals, rewards, alpha=0.3, where='post', linestyle='--', label=f"{method} Seed {j+1}" if j == 0 else None)
        
#         # Interpolated mean curve
#         interp_curves = []
#         for curve in curves:
#             if not curve:
#                 continue
#             rewards, evals = zip(*curve)
#             interp = np.interp(x_eval, evals, rewards, left=rewards[0], right=rewards[-1])
#             interp_curves.append(interp)
#             print(f"INTERP {method}: {interp[:5]}... to {interp[-5:]}")
        
#         if not interp_curves:
#             continue
#         interp_curves = np.array(interp_curves)
#         mean_curve = interp_curves.mean(axis=0)
#         se_curve = interp_curves.std(axis=0, ddof=1) / np.sqrt(num_seeds)
#         print(f"{method} mean: {mean_curve[:5]}... to {mean_curve[-5:]}")
#         plt.plot(x_eval, mean_curve, label=method, linewidth=2)
#         plt.fill_between(x_eval, mean_curve - 1.96 * se_curve, mean_curve + 1.96 * se_curve, alpha=0.2)
    
#     plt.xscale('log')
#     plt.xlabel("Evaluations")
#     plt.ylabel("Best Return")
#     plt.title(task)
#     plt.legend()
#     plt.grid(True, which="both")
#     # Set y-axis limits to capture full reward range
#     all_rewards = [r for method in methods for curve in results[task][method]["curves"] for r, _ in curve if curve]
#     if all_rewards:
#         plt.ylim(min(-0.5, min(all_rewards)), max(1.1, max(all_rewards)))
#         print(f"Y-limits for {task}: {min(-0.5, min(all_rewards))} to {max(1.1, max(all_rewards))}")
#     else:
#         plt.ylim(-0.5, 1.1)

# plt.tight_layout()
# plt.savefig("figure3_convergence.png")
# plt.show()

# # Figure 3
# plt.figure(figsize=(15, 10))
# for i, task in enumerate(all_tasks):
#     plt.subplot(4, 3, i + 1)
#     print(f"\nProcessing task: {task}")

#     for method in methods:
#         curves = results[task][method]["curves"]

#         if not curves:
#             print(f"No curves for {method} in {task}")
#             continue

#         max_evals = max(max(curve[-1][1] for curve in curves if curve), 1)
#         x_eval = np.logspace(0, np.log10(max_evals), 100)
#         print(f"{method}: max_evals={max_evals}, x_eval range={x_eval[0]} to {x_eval[-1]}")

#         interp_curves = []
        
#         # Plot raw step curves for each seed (faint)
#         for curve in curves:
#             if not curve:
#                 continue
#             evals, rewards = zip(*curve)
#             print('CURVE')
#             print(evals, rewards)
#             plt.step(evals, rewards, alpha=0.3, where='post', linestyle='--')
        
#         # Interpolated mean curve
#         for curve in curves:
#             if not curve:
#                 continue
#             evals, rewards = zip(*curve)
#             # Use first reward as left fill to avoid -inf skew
#             interp = np.interp(x_eval, evals, rewards, left=rewards[0], right=rewards[-1])
#             interp_curves.append(interp)
#             print('INTERP')
#             print(interp)
        
#         if not interp_curves:
#             continue
#         interp_curves = np.array(interp_curves)
#         mean_curve = interp_curves.mean(axis=0)
#         se_curve = interp_curves.std(axis=0, ddof=1) / np.sqrt(num_seeds)
#         plt.plot(x_eval, mean_curve, label=method, linewidth=2)
#         plt.fill_between(x_eval, mean_curve - 1.96 * se_curve, mean_curve + 1.96 * se_curve, alpha=0.2)
    
#     plt.xscale('log')
#     plt.xlabel("Evaluations")
#     plt.ylabel("Best Return")
#     plt.title(task)
#     plt.legend()
#     plt.grid(True, which="both")
#     # Set y-axis limits to capture full reward range
#     plt.ylim(min(-0.5, min(r for curve in results[task][method]["curves"] for r, _ in curve if curve)),
#              max(1.1, max(r for curve in results[task][method]["curves"] for r, _ in curve if curve)))
# plt.tight_layout()
# plt.savefig("figure3_convergence.png")
# plt.show()

# # Figure 3
# plt.figure(figsize=(15, 10))
# for i, task in enumerate(all_tasks):
#     plt.subplot(4, 3, i + 1)
#     for method in methods:
#         curves = results[task][method]["curves"]
#         max_evals = max(max(curve[-1][1] for curve in curves if curve), 1)
#         x_eval = np.logspace(0, np.log10(max_evals), 100)
#         interp_curves = []
#         for curve in curves:
#             if not curve:  # Handle empty curves
#                 continue
#             evals, rewards = zip(*curve)
#             interp = np.interp(x_eval, evals, rewards, left=-float('inf'), right=rewards[-1])
#             interp_curves.append(interp)
#         if not interp_curves:  # Skip if no valid curves
#             continue
#         interp_curves = np.array(interp_curves)
#         mean_curve = interp_curves.mean(axis=0)
#         se_curve = interp_curves.std(axis=0, ddof=1) / np.sqrt(num_seeds)
#         plt.plot(x_eval, mean_curve, label=method)
#         plt.fill_between(x_eval, mean_curve - 1.96 * se_curve, mean_curve + 1.96 * se_curve, alpha=0.2)
#     plt.xscale('log')
#     plt.xlabel("Evaluations")
#     plt.ylabel("Best Return")
#     plt.title(task)
#     plt.legend()
#     plt.grid(True, which="both")
# plt.tight_layout()
# plt.savefig("figure3_convergence.png")
# plt.show()

# num_seeds = 5

# # Containers for results over seeds
# final_returns = []
# eval_counts = []
# reward_curves = []  # list of lists: best reward per iteration for each run

# # Loop over independent seeds
# for seed in range(num_seeds):
#     # Set the random seeds for reproducibility
#     Config.env_seed = seed
#     Config.model_seed = seed
#     torch.manual_seed(seed)
#     np.random.seed(seed)
    
#     # Initialize DSL, device, and model
#     dsl = DSL.init_default_karel()
#     device = torch.device('cpu') if Config.disable_gpu else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = load_model(Config.model_name, dsl, device)
#     task_cls = get_task_cls(Config.env_task)
#     params = torch.load(Config.model_params_path, map_location=device)
#     model.load_state_dict(params, strict=False)
#     model.eval()
    
#     # Create a new LatentSearch object for this seed
#     searcher = LatentSearch(model, task_cls, dsl)
    
#     StdoutLogger.log('Main', f'Run {seed+1}/{num_seeds}: Starting Latent Search with model {Config.model_name} for task {Config.env_task}')
    
#     # Run the search and collect results.
#     # (Assume that search() has been modified to return: best_program, converged, num_evaluations, reward_curve)
#     best_program, converged, num_evals, reward_curve = searcher.search(return_curve=True)
    
#     final_returns.append(searcher.best_reward)
#     eval_counts.append(num_evals)
#     reward_curves.append(reward_curve)
    
#     StdoutLogger.log('Main', f'Run {seed+1}: Converged: {converged}, Final Reward: {searcher.best_reward}, Evaluations: {num_evals}')

# # Convert final returns to a NumPy array for statistical analysis
# final_returns = np.array(final_returns)
# mean_return = final_returns.mean()
# std_return = final_returns.std(ddof=1)
# se_return = std_return / np.sqrt(num_seeds)
# ci_return = (mean_return - 1.96 * se_return, mean_return + 1.96 * se_return)

# print("Final Episodic Return over {} seeds: Mean = {:.3f}, 95% CI = ({:.3f}, {:.3f})".format(
#     num_seeds, mean_return, ci_return[0], ci_return[1]))

# # Optionally, print number of evaluations statistics
# eval_counts = np.array(eval_counts)
# mean_evals = eval_counts.mean()
# print("Mean number of evaluations: {:.0f}".format(mean_evals))

# # ---------- Plotting Convergence Curves ----------
# # First, you need to align the curves.
# # For example, assume that each reward_curve is a list of best reward values recorded per iteration.
# # You can plot each curve separately, or compute an average curve.
# max_iters = max(len(curve) for curve in reward_curves)
# # Interpolate each curve to the same number of points if needed:
# aligned_curves = []
# for curve in reward_curves:
#     if len(curve) < max_iters:
#         # Pad with the last value
#         curve = curve + [curve[-1]] * (max_iters - len(curve))
#     aligned_curves.append(curve)
# aligned_curves = np.array(aligned_curves)  # shape: (num_seeds, max_iters)

# # Compute the mean and standard error across seeds
# mean_curve = aligned_curves.mean(axis=0)
# std_curve = aligned_curves.std(axis=0, ddof=1)
# se_curve = std_curve / np.sqrt(num_seeds)
# # Define x-axis: iterations (or evaluations). If you record evaluations per iteration, you can use that.
# x = np.arange(1, max_iters + 1)

# plt.figure(figsize=(10, 6))
# plt.errorbar(x, mean_curve, yerr=1.96 * se_curve, fmt='-o', capsize=4)
# plt.xlabel("Iteration")
# plt.ylabel("Best Episodic Return")
# plt.title("Convergence Curve of Latent Search (Reward vs. Iteration)")
# plt.grid(True)
# plt.show()

# # If you wish to plot episodic return vs. number of evaluations on a log scale,
# # you need to record the evaluation count at each iteration. For example, if each iteration takes a fixed number of evaluations,
# # you can compute x_eval = iterations * (population_size) or use recorded counts.
# # For illustration, assuming each iteration uses population_size evaluations:
# x_eval = x * Config.search_population_size
# plt.figure(figsize=(10, 6))
# plt.errorbar(x_eval, mean_curve, yerr=1.96 * se_curve, fmt='-o', capsize=4)
# plt.xlabel("Number of Evaluations (log scale)")
# plt.xscale("log")
# plt.ylabel("Best Episodic Return")
# plt.title("Convergence Curve of Latent Search (Reward vs. Evaluations)")
# plt.grid(True, which="both")
# plt.show()

# # ---------- Generate a Table of Final Results ----------
# import pandas as pd

# results_df = pd.DataFrame({
#     "Seed": np.arange(1, num_seeds+1),
#     "Final Return": final_returns,
#     "Evaluations": eval_counts
# })
# print(results_df.describe())

# # You can also save this table as CSV:
# results_df.to_csv("latent_search_final_results.csv", index=False)
