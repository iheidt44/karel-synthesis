import torch
import numpy as np
import matplotlib.pyplot as plt
from vae.models import load_model
from dsl import DSL
from dsl.base import Program
from config import Config
from karel.world_generator import WorldGenerator
from tasks.task import Task
import sys

np.set_printoptions(threshold=np.inf)

Config.model_hidden_size = 256
# in paper 32
num_maps = 8
# in paper 1000
population_size = 100
num_iters = 10
# sigmas = [0.1, 0.25, 0.5]
# sigmas = [0.75, 1.0, 1.25]
sigmas = [1.5]

sigma_colors = {
    0.1: 'red',
    0.25: 'blue',
    0.5: 'green'
}

def longest_common_prefix(seq1, seq2):
    lcp = 0
    alist = []
    for a, b in zip(seq1, seq2):
        if a == b:
            lcp += 1
        else:
            alist.append(lcp)
            lcp = 0
    alist.append(lcp)
    # print(alist)
    return max(alist)

def prepare_s_h(world, model, population_size):
    state_tensor = torch.tensor(world.get_state(), dtype=torch.float32, device=device)
    state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0) 
    s_h = state_tensor.repeat(population_size, 1, 1, 1, 1)  
    s_h = s_h.unsqueeze(2).repeat(1, 1, model.max_demo_length, 1, 1, 1)
    return s_h

class GenericKarelTask(Task):
    def generate_state(self):
        wg = WorldGenerator()
        return wg.generate(wall_prob=0.1, marker_prob=0.1)
    
    def get_reward(self, world_state):
        return False, 0.0

def get_action_sequence(program, env):
    env.reset_state()
    actions = []
    count = 0
    for node in program.run_generator(env.get_state()):
        if node is not None and node.__class__.__name__ in ["Move", "TurnLeft", "TurnRight", "PickMarker", "PutMarker"]:
            action_idx = dsl.a2i[node.__class__]
            actions.append(action_idx)
            count += 1
        if env.get_state().is_crashed() or count == model.max_demo_length:
            break
    return np.array(actions)


def decode_complete_program(model, z, dsl, max_attempts=10):
    for _ in range(max_attempts):
        with torch.no_grad():
            tokens = model.decode_vector(z.unsqueeze(0))[0] 
            try:
                program = dsl.parse_int_to_node(tokens)
                if program.is_complete():
                    return program
            except AssertionError:
                continue 
    return Program.new(dsl.parse_str_to_node("move"))

dsl = DSL.init_default_karel()
device = torch.device('cpu')
task_envs = [GenericKarelTask(i) for i in range(num_maps)]

models = {
    "NewLatent": {"path": "output/semantic_only_256_v2/model/best_val.ptp"},
    "LEAPS": {"path": "params/leaps_vae_256.ptp"}
}

for name in models:
    model = load_model("LeapsVAE", dsl, device)
    params = torch.load(models[name]["path"], map_location=device)
    model.load_state_dict(params, strict=False)
    model.eval()
    models[name]["model"] = model

s_h_list = [prepare_s_h(env.get_state(), models["NewLatent"]["model"], population_size) for env in task_envs]

results = {"NewLatent": {}, "LEAPS": {}}

for model_name in models:
    model = models[model_name]["model"]
    latent_dim = model.hidden_size
    for sigma in sigmas:
        print(f"Evalutating model {model_name} simga = {sigma}")
        behavior_metrics = []
        identity_metrics = []
        current_latents = torch.randn(population_size, latent_dim, device=device)

        baseline_actions_list = []
        with torch.no_grad():
            if model_name == "NewLatent":
                for s_h in s_h_list:
                    a_h = torch.full((population_size, 1, model.max_demo_length), fill_value=5, dtype=torch.long, device=device)
                    a_h_masks = torch.zeros_like(a_h, dtype=torch.bool, device=device)
                    pred_actions, _, _ = model.policy_executor(current_latents, s_h, a_h, a_h_masks, a_h_teacher_enforcing=False)
                    baseline_actions_list.append(pred_actions.cpu().numpy())
            else: 
                programs = [decode_complete_program(model, z, dsl) for z in current_latents]
                baseline_actions_list = [[get_action_sequence(prog, env) for env in task_envs] for prog in programs]

        for iter in range(num_iters):
            noise = sigma * torch.randn_like(current_latents, device=device)
            new_latents = current_latents + noise
            new_actions_list = []
            with torch.no_grad():
                if model_name == "NewLatent":
                    for s_h in s_h_list:
                        a_h = torch.full((population_size, 1, model.max_demo_length), fill_value=5, dtype=torch.long, device=device)
                        a_h_masks = torch.zeros_like(a_h, dtype=torch.bool, device=device)
                        pred_actions, _, _ = model.policy_executor(new_latents, s_h, a_h, a_h_masks, a_h_teacher_enforcing=False)
                        new_actions_list.append(pred_actions.cpu().numpy())
                else: 
                    programs = [decode_complete_program(model, z, dsl) for z in new_latents]
                    new_actions_list = [[get_action_sequence(prog, env) for env in task_envs] for prog in programs]

            seed_similarities = []
            seed_identities = []
            for i in range(population_size):
                sim_values = []
                id_values = []
                for env_idx in range(num_maps):
                    baseline_seq = baseline_actions_list[env_idx][i] if model_name == "NewLatent" else baseline_actions_list[i][env_idx]
                    new_seq = new_actions_list[env_idx][i] if model_name == "NewLatent" else new_actions_list[i][env_idx]
                    
                    baseline_seq_no_nop = baseline_seq[baseline_seq != 5]
                    new_seq_no_nop = new_seq[new_seq != 5]
                    
                    print("Testing baseline compared to new:")
                    print("Baseline: ", baseline_seq)
                    print("New: ", new_seq)
                    lcp = longest_common_prefix(baseline_seq_no_nop, new_seq_no_nop)
                    print("Found LCP: ", lcp)
                    max_len = max(len(baseline_seq_no_nop), len(new_seq_no_nop))
                    if max_len > 0:
                        num = lcp / max_len
                    else:
                        num = 0
                    print('Normalized lcp: ', num)
                    sim_values.append(num)
                    id_values.append(1 if np.array_equal(baseline_seq_no_nop, new_seq_no_nop) else 0)

                seed_similarities.append(np.mean(sim_values))
                seed_identities.append(np.mean(id_values))

            mean_sim, se_sim = np.mean(seed_similarities), np.std(seed_similarities, ddof=1) / np.sqrt(population_size)
            mean_id, se_id = np.mean(seed_identities), np.std(seed_identities, ddof=1) / np.sqrt(population_size)
            behavior_metrics.append((mean_sim, mean_sim - 1.96 * se_sim, mean_sim + 1.96 * se_sim))
            identity_metrics.append((mean_id, mean_id - 1.96 * se_id, mean_id + 1.96 * se_id))

            print(f"{model_name}, Sigma={sigma}, Iteration {iter+1}: Behavior={mean_sim:.3f}, Identity={mean_id:.3f}\n\n\n")
            current_latents = new_latents

        results[model_name][sigma] = {"behavior": behavior_metrics, "identity": identity_metrics}


for model_name in results:
    for sigma in sigmas:
        behavior_file = f"behaviour_{model_name}_{sigma}.txt"
        with open(behavior_file, 'w') as f:
            f.write("Iteration,Mean,Lower_CI,Upper_CI\n")
            for i, (mean, lower, upper) in enumerate(results[model_name][sigma]["behavior"], 1):
                f.write(f"{i},{mean:.6f},{lower:.6f},{upper:.6f}\n")
        
        identity_file = f"identity_{model_name}_{sigma}.txt"
        with open(identity_file, 'w') as f:
            f.write("Iteration,Mean,Lower_CI,Upper_CI\n")
            for i, (mean, lower, upper) in enumerate(results[model_name][sigma]["identity"], 1):
                f.write(f"{i},{mean:.6f},{lower:.6f},{upper:.6f}\n")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for model_name in results:
    for sigma in sigmas:
        behavior = results[model_name][sigma]["behavior"]
        mean = [m[0] for m in behavior]
        lower = [m[1] for m in behavior]
        upper = [m[2] for m in behavior]
        x = range(1, num_iters + 1)
        label = f"{model_name} σ={sigma}"
        color = sigma_colors[sigma]
        linestyle = '--' if model_name == "NewLatent" else '-'
        plt.plot(x, mean, label=label, color=color, linestyle=linestyle)
        plt.fill_between(x, lower, upper, alpha=0.2, color=color)
plt.xlabel("Number of Mutations")
plt.ylabel("Behavior Similarity")
plt.legend()
plt.title("Behavior Similarity")
plt.ylim(0, 1)
plt.xlim(1, 10)

plt.subplot(1, 2, 2)
for model_name in results:
    for sigma in sigmas:
        identity = results[model_name][sigma]["identity"]
        mean = [m[0] for m in identity]
        lower = [m[1] for m in identity]
        upper = [m[2] for m in identity]
        x = range(1, num_iters + 1)
        label = f"{model_name} σ={sigma}"
        color = sigma_colors[sigma]
        linestyle = '--' if model_name == "NewLatent" else '-'
        plt.plot(x, mean, label=label, color=color, linestyle=linestyle)
        plt.fill_between(x, lower, upper, alpha=0.2, color=color)

plt.xlabel("Number of Mutations")
plt.ylabel("Identity Rate")
plt.legend()
plt.title("Identity Rate")
plt.ylim(0, 1)
plt.xlim(1, 10)
# plt.tight_layout()
plt.savefig("figure4_latent_only.png")
plt.show()