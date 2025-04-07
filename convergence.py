import torch
import numpy as np
import matplotlib.pyplot as plt
from vae.models import load_model
from dsl import DSL
from config import Config
from tasks import get_task_cls  # Assumes you have a function to get your task class
from karel.world_generator import WorldGenerator

# ---------- Helper Functions ----------

def prepare_single_s_h(world):
    """
    Prepares an s_h tensor for a single candidate from a given world.
    Output shape: [1, 1, max_demo_length, channels, height, width]
    """
    state_tensor = torch.tensor(world.get_state(), dtype=torch.float32, device=device)
    # Permute to (channels, H, W) and add batch dimension
    state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    # Add demos_per_program dimension (1) and time dimension (max_demo_length)
    s_h = state_tensor.unsqueeze(1)            # [1, 1, C, H, W]
    s_h = s_h.unsqueeze(2).repeat(1, 1, model.max_demo_length, 1, 1, 1)  # [1, 1, max_demo_length, C, H, W]
    return s_h

def evaluate_return(latent, task):
    """
    Given a latent vector and a task instance (which can generate an initial state),
    this function runs the policy decoded from the latent on the task’s initial state
    and returns the episodic return.
    """
    # Get the initial state from the task (assumes task.generate_state() returns a world-like object)
    state = task.generate_state()
    task.reset_state()  # reset the task/environment
    
    # Prepare s_h for a single candidate
    s_h = prepare_single_s_h(state)  # shape: [1, 1, max_demo_length, C, H, W]
    
    # Prepare dummy action history and mask (for a single candidate)
    a_h = torch.full((1, 1, model.max_demo_length),
                     fill_value=model.num_agent_actions - 1,
                     dtype=torch.long, device=device)
    a_h_masks = torch.zeros_like(a_h, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        # latent is a vector of shape [latent_dim]; unsqueeze to [1, latent_dim]
        pred_actions, _, _ = model.policy_executor(latent.unsqueeze(0), s_h, a_h, a_h_masks,
                                                    a_h_teacher_enforcing=False)
    # Get predicted actions (shape: [max_demo_length])
    actions = pred_actions.cpu().numpy()[0]
    
    total_reward = 0.0
    # Simulate execution in the task/environment:
    for action in actions:
        # If action is NOP (assumed to be represented as num_agent_actions - 1), terminate execution.
        if action >= model.num_agent_actions - 1:
            break
        task.state.run_action(action)
        terminated, r = task.get_reward(task.get_state())
        total_reward += r
        if terminated or task.state.is_crashed():
            break
    return total_reward

def hill_climb(latent, task, max_iters=10, sigma=0.25):
    """
    Runs hill climbing on a single candidate latent vector for a given task.
    Returns the best episodic return found.
    """
    current_latent = latent.clone()
    best_return = evaluate_return(current_latent, task)
    for _ in range(max_iters):
        noise = sigma * torch.randn_like(current_latent)
        new_latent = current_latent + noise
        new_return = evaluate_return(new_latent, task)
        if new_return > best_return:
            current_latent = new_latent
            best_return = new_return
        # else:
        #     # Stop if no improvement is found (local maximum reached)
        #     break
    return best_return

# ---------- Convergence Analysis Setup ----------

# Parameters for convergence analysis
N_seeds = 100       # Number of initial latent candidates (P0)
N_states = 8         # Number of initial states (S0)
#gtarget = 0.5          # Target episodic return for convergence
g_target_range = np.linspace(0, 1, 101)
max_iters = 10         # Maximum iterations for hill climbing per candidate
sigma_hc = 0.25        # Noise level used in hill climbing
Config.model_hidden_size = 64

tasks = ["StairClimber", "Maze", "FourCorners", 
         "Harvester", "CleanHouse", "TopOff",
         "DoorKey", "FindMarker", "Seeder", "OneStroke", "Snake"]

dsl = DSL.init_default_karel()
device = torch.device('cpu')
model = load_model("LeapsVAE", dsl, device)
params = torch.load("output/semantic_only_64_mar/model/best_val.ptp", map_location=device)
model.load_state_dict(params, strict=False)
model.eval()

# Sample N_seeds latent vectors from N(0, I)
latent_dim = model.hidden_size  # or model.hidden_size
seeds = torch.randn(N_seeds, latent_dim, device=device)

# Create a set of task instances (each with its own initial state) from S0.
# Here we assume get_task_cls returns a Task class that can be instantiated.
Config.env_task = tasks[0]
TaskCls = get_task_cls(Config.env_task)
tasks = [TaskCls(i) for i in range(N_states)]

best_returns = np.zeros((N_states, N_seeds))
for s_idx, task in enumerate(tasks):
    print(f"Evaluating task {s_idx+1}/{N_states}...")
    for i in range(N_seeds):
        best_returns[s_idx, i] = hill_climb(seeds[i], task, max_iters=max_iters, sigma=sigma_hc)

convergence_rates = []  # mean convergence rate for each g_target
convergence_rates_ci = []  # 95% confidence intervals (as a tuple: (lower, upper))

for g in g_target_range:
    rates_per_state = (best_returns >= g).mean(axis=1)  # fraction for each state
    mean_rate = rates_per_state.mean()
    std_rate = rates_per_state.std(ddof=1)
    se_rate = std_rate / np.sqrt(N_states)
    ci = (mean_rate - 1.96 * se_rate, mean_rate + 1.96 * se_rate)
    convergence_rates.append(mean_rate)
    convergence_rates_ci.append(ci)

# Plot the convergence rate curve with error bars (95% CI)
plt.figure(figsize=(8, 6))
plt.errorbar(g_target_range, convergence_rates,
             yerr=[(mean - ci[0]) for mean, ci in zip(convergence_rates, convergence_rates_ci)],
             fmt='-o', capsize=4)
plt.xlabel("g_target")
plt.ylabel("Convergence Rate")
plt.title("Convergence Rate vs. g_target")
plt.grid(True)
plt.show()

# ---------- Run Hill Climbing for Each (Seed, Task) Pair ----------

# # We'll record a binary outcome for each pair: 1 if hill climbing returns >= gtarget, else 0.
# results = []  # will be a list of length N_states, each element is an array of outcomes for N_seeds

# for task in tasks:
#     outcomes = []  # outcomes for current task (over all seeds)
#     print(f"Evaluating task with initial seed {task.seed if hasattr(task, 'seed') else 'N/A'}...")
#     for i in range(N_seeds):
#         best_return = hill_climb(seeds[i], task, max_iters=max_iters, sigma=sigma_hc)
#         outcomes.append(1 if best_return >= gtarget else 0)
#     outcomes = np.array(outcomes)
#     convergence_rate = outcomes.mean()  # fraction of seeds that converge to return >= gtarget for this task
#     results.append(convergence_rate)
#     print(f"Task convergence rate: {convergence_rate:.3f}")

# # ---------- Aggregate Convergence Rate Across Tasks (States) ----------

# results = np.array(results)  # shape: (N_states,)
# overall_convergence_rate = results.mean()
# std_rate = results.std(ddof=1)
# se_rate = std_rate / np.sqrt(N_states)
# ci_rate = (overall_convergence_rate - 1.96 * se_rate, overall_convergence_rate + 1.96 * se_rate)

# print("\nConvergence Analysis Results:")
# print(f"Overall convergence rate (g_target = {gtarget}): {overall_convergence_rate:.3f}")
# print(f"95% confidence interval: ({ci_rate[0]:.3f}, {ci_rate[1]:.3f})")
