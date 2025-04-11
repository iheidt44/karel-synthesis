import torch
import numpy as np
import matplotlib.pyplot as plt
from vae.models import load_model
from dsl import DSL
from config import Config
from tasks import get_task_cls
from search.top_down import TopDownSearch
from dsl.base import Program
from tasks.task import Task
from karel.world_generator import WorldGenerator
import csv
import copy

np.set_printoptions(threshold=np.inf)

def prepare_s_h(world, model, population_size):
    state_tensor = torch.tensor(world.get_state(), dtype=torch.float32, device=device)
    state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)  
    s_h = state_tensor.repeat(population_size, 1, 1, 1, 1)
    s_h = s_h.unsqueeze(2).repeat(1, 1, Config.data_max_demo_length, 1, 1, 1)
    return s_h


def execute_program_new_latent_batch(latents, task_envs, model):

    K = latents.shape[0] 
    num_envs = len(task_envs)
    task_envs = [copy.deepcopy(env) for env in original_task_envs]
    rewards = torch.zeros(K, device=device)

    all_actions = []
    for env in task_envs:
        s_h = prepare_s_h(env.get_state(), model, K) 
        a_h = torch.full((K, 1, Config.data_max_demo_length), fill_value=model.num_agent_actions - 1,
                         dtype=torch.long, device=device)
        a_h_masks = torch.zeros_like(a_h, dtype=torch.bool, device=device)
        with torch.no_grad():
            pred_actions, _, _ = model.policy_executor(latents, s_h, a_h, a_h_masks,
                                                      a_h_teacher_enforcing=False)
        all_actions.append(pred_actions.cpu().numpy())  

    for i in range(K):
        total_reward = 0.0
        for env_idx, env in enumerate(task_envs):
            env = copy.deepcopy(original_task_envs[env_idx])
            for action in all_actions[env_idx][i]:
                if action >= model.num_agent_actions - 1:
                    break
                env.state.run_action(action)
                terminated, r = env.get_reward(env.get_state())
                total_reward += r
                if terminated or env.state.is_crashed():
                    break
        rewards[i] = total_reward / num_envs
    return rewards

def execute_program_leaps(latent, task_envs, model, dsl):
    task_envs = [copy.deepcopy(env) for env in original_task_envs]
    with torch.no_grad():
        programs_tokens = model.decode_vector(latent.unsqueeze(0))[0]
        programs_str = dsl.parse_int_to_str(programs_tokens)

        try:
            program = dsl.parse_str_to_node(programs_str)
        except AssertionError:
            print('Invalid prog')
            return -float('inf')
    
    if not program.is_complete():
        tds = TopDownSearch()
        tds_result = tds.synthesize(program, dsl, task_envs, Config.datagen_sketch_iterations)
        program, _, mean_reward = tds_result
        if program is None or not program.is_complete():
            return -float('inf')
    else:
        mean_reward = 0.0
        for task_env in task_envs:
            mean_reward += task_env.evaluate_program(program)
        mean_reward /= len(task_envs)
    
    return mean_reward

def hill_climb(latent, task_envs, model, dsl, model_name, K=10, sigma=0.25):
    current_latent = latent.clone()
    if model_name == "NewLatent":
        best_return = execute_program_new_latent_batch(current_latent.unsqueeze(0), task_envs, model)[0]
    else:
        best_return = execute_program_leaps(current_latent, task_envs, model, dsl)
    
    while True:
        noise = sigma * torch.randn(K, latent.size(0), device=device)
        neighbors = current_latent.unsqueeze(0) + noise

        if model_name == "NewLatent":
            returns = execute_program_new_latent_batch(neighbors, task_envs, model)
        else: 
            returns = torch.tensor([execute_program_leaps(n, task_envs, model, dsl) for n in neighbors], device=device)
        
        best_neighbor_idx = torch.argmax(returns)
        best_neighbor_return = returns[best_neighbor_idx]
        
        if best_neighbor_return > best_return:
            current_latent = neighbors[best_neighbor_idx]
            best_return = best_neighbor_return
        else:
            break
    
    return best_return

# initial candidate latent vectors, 10,000 in paper
N_seeds = 10
# initial env states, 32 in paper
N_states = 4
g_target_range = np.linspace(0, 1, 101)

# sigmas = {"StairClimber" : 0.25, 
#           "Maze" : 0.75, 
#           "FourCorners" : 0.5, 
#           "Harvester" : 0.5, 
#           "TopOff" : 0.25,
#           "DoorKey" : 0.25, 
#           "Seeder" : 0.25, 
#           "OneStroke" : 0.25, 
#           "Snake" : 0.25
# }

sigmas = {"StairClimber" : [0.25, 0.5, 0.75, 1.0], 
          "Maze" : [0.1, 0.5, 0.75, 1.0], 
          "FourCorners" : [0.5, 0.75, 1.0], 
          "Harvester" : [0.5, 0.75, 1.0], 
          "TopOff" : [0.25, 0.5, 0.75, 1.0],
          "DoorKey" : [0.25, 0.5, 0.75, 1.0], 
          "Seeder" : [0.25, 0.5, 0.75, 1.0], 
          "OneStroke" : [0.25, 0.5, 0.75, 1.0], 
          "Snake" : [0.25, 0.5, 0.75, 1.0]
}
# neighbourhood size, 250 in paper
K_hc = 250
Config.model_hidden_size = 256
Config.data_max_demo_length = 300
Config.env_enable_leaps_behaviour = True
Config.disable_gpu = True

dsl = DSL.init_default_karel()
device = torch.device('cpu')

models = {
    "NewLatent": {"path": "output/semantic_only_256_v2/model/best_val.ptp"},
    "LEAPS": {"path": "params/leaps_vae_256.ptp"}
}

# models = {
#     "LEAPS": {"path": "params/leaps_vae_256.ptp"}
# }

for name in models:
    model = load_model("LeapsVAE", dsl, device)
    params = torch.load(models[name]["path"], map_location=device)
    model.load_state_dict(params, strict=False)
    model.eval()
    models[name]["model"] = model

latent_dim = Config.model_hidden_size
seeds = torch.randn(N_seeds, latent_dim, device=device)

tasks = ["StairClimber", "Maze", "FourCorners", "Harvester", "TopOff",
         "DoorKey", "Seeder", "OneStroke", "Snake"]
results = {"NewLatent": {}, "LEAPS": {}}

# tasks = ["Maze"]
# results = {"LEAPS": {}}

for task_name in tasks:
    print(f"\nProcessing task: {task_name}")
    Config.env_task = task_name
    TaskCls = get_task_cls(Config.env_task)
    original_task_envs = [TaskCls(i) for i in range(N_states)]

    for sigma_hc in sigmas[task_name]:
        print(f" Testing sigma: {sigma_hc}")

        for model_name in models:
            model = models[model_name]["model"]
            print(f"  Evaluating {model_name}...")
            best_returns = np.zeros((N_states, N_seeds))
            for s_idx in range(N_states):
                print(f"    State {s_idx+1}/{N_states}")
                for i in range(N_seeds):
                    print(f"       Seed {i+1}/{N_seeds}")
                    best_returns[s_idx, i] = hill_climb(seeds[i], original_task_envs, model, dsl,
                                                        model_name, K_hc, sigma_hc)

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
            
            if task_name not in results[model_name]:
                results[model_name][task_name] = {}

            results[model_name][task_name][sigma_hc] = {
                "rates": convergence_rates,
                "ci": convergence_rates_ci
            }

            filename = f"convergence_{model_name}_{task_name}_{sigma_hc}.csv"
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["g_target", "Mean", "Lower_CI", "Upper_CI"])
                for g, mean, (lower, upper) in zip(g_target_range, convergence_rates, convergence_rates_ci):
                    writer.writerow([g, mean, lower, upper])
            print(f"  Saved data to {filename}")


sigma_colors = {"NewLatent": 'red', "LEAPS": 'blue'}
for task_name in tasks:
    num_sigmas = len(sigmas[task_name])
    plt.figure(figsize=(5 * num_sigmas, 5))
    for idx, sigma_hc in enumerate(sigmas[task_name], 1):
        plt.subplot(1, num_sigmas, idx)
        for model_name in models:
            rates = results[model_name][task_name][sigma_hc]["rates"]
            ci = results[model_name][task_name][sigma_hc]["ci"]
            mean = rates
            lower = [c[0] for c in ci]
            upper = [c[1] for c in ci]
            x = g_target_range
            label = model_name
            color = sigma_colors[model_name]
            linestyle = '--' if model_name == "NewLatent" else '-'
            plt.plot(x, mean, label=label, color=color, linestyle=linestyle)
            if N_states > 1:
                plt.fill_between(x, lower, upper, alpha=0.2, color=color)
        plt.xlabel("g_target")
        plt.ylabel("Convergence Rate")
        plt.title(f"{task_name}-{sigma_hc}")
        plt.ylim(0, 1)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"convergence_{task_name}.png")
    plt.show()