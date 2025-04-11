import torch
import numpy as np
import matplotlib.pyplot as plt
from vae.models import load_model
from dsl import DSL
from config import Config
from tasks import get_task_cls
from karel.world_generator import WorldGenerator

def prepare_single_s_h(world):
    state_tensor = torch.tensor(world.get_state(), dtype=torch.float32, device=device)
    state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0) 
    s_h = state_tensor.unsqueeze(1)       
    s_h = s_h.unsqueeze(2).repeat(1, 1, model.max_demo_length, 1, 1, 1) 
    return s_h

def evaluate_return(latent, task):
    state = task.generate_state()
    task.reset_state()

    s_h = prepare_single_s_h(state)  
    
    a_h = torch.full((1, 1, model.max_demo_length),
                     fill_value=model.num_agent_actions - 1,
                     dtype=torch.long, device=device)
    a_h_masks = torch.zeros_like(a_h, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        pred_actions, _, _ = model.policy_executor(latent.unsqueeze(0), s_h, a_h, a_h_masks,
                                                    a_h_teacher_enforcing=False)
    actions = pred_actions.cpu().numpy()[0]
    
    total_reward = 0.0
    for action in actions:
        if action >= model.num_agent_actions - 1:
            break
        task.state.run_action(action)
        terminated, r = task.get_reward(task.get_state())
        total_reward += r
        if terminated or task.state.is_crashed():
            break
    return total_reward

def hill_climb(latent, task, max_iters=10, sigma=0.25):
    current_latent = latent.clone()
    best_return = evaluate_return(current_latent, task)
    for _ in range(max_iters):
        noise = sigma * torch.randn_like(current_latent)
        new_latent = current_latent + noise
        new_return = evaluate_return(new_latent, task)
        if new_return > best_return:
            current_latent = new_latent
            best_return = new_return
    return best_return

N_seeds = 100  
N_states = 8     
g_target_range = np.linspace(0, 1, 101)
max_iters = 10 
sigma_hc = 0.25     
Config.model_hidden_size = 256

tasks = ["StairClimber", "Maze", "FourCorners", 
         "Harvester", "CleanHouse", "TopOff",
         "DoorKey", "FindMarker", "Seeder", "OneStroke", "Snake"]

dsl = DSL.init_default_karel()
device = torch.device('cpu')
model = load_model("LeapsVAE", dsl, device)
params = torch.load("output/semantic_only_256_v2/model/best_val.ptp", map_location=device)
model.load_state_dict(params, strict=False)
model.eval()

latent_dim = model.hidden_size 
seeds = torch.randn(N_seeds, latent_dim, device=device)

Config.env_task = tasks[0]
TaskCls = get_task_cls(Config.env_task)
tasks = [TaskCls(i) for i in range(N_states)]

best_returns = np.zeros((N_states, N_seeds))
for s_idx, task in enumerate(tasks):
    print(f"Evaluating task {s_idx+1}/{N_states}...")
    for i in range(N_seeds):
        best_returns[s_idx, i] = hill_climb(seeds[i], task, max_iters=max_iters, sigma=sigma_hc)

convergence_rates = [] 
convergence_rates_ci = [] 

for g in g_target_range:
    rates_per_state = (best_returns >= g).mean(axis=1)
    mean_rate = rates_per_state.mean()
    std_rate = rates_per_state.std(ddof=1)
    se_rate = std_rate / np.sqrt(N_states)
    ci = (mean_rate - 1.96 * se_rate, mean_rate + 1.96 * se_rate)
    convergence_rates.append(mean_rate)
    convergence_rates_ci.append(ci)

plt.figure(figsize=(8, 6))
plt.errorbar(g_target_range, convergence_rates,
             yerr=[(mean - ci[0]) for mean, ci in zip(convergence_rates, convergence_rates_ci)],
             fmt='-o', capsize=4)
plt.xlabel("g_target")
plt.ylabel("Convergence Rate")
plt.title("Convergence Rate vs. g_target")
plt.grid(True)
plt.show()
