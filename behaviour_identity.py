import torch
import numpy as np
from vae.models import load_model
from dsl import DSL
from config import Config
from tasks import get_task_cls
from karel.world_generator import WorldGenerator

def prepare_s_h(world):
    """
    Converts a world into an s_h tensor of shape:
      (population_size, demos_per_program, max_demo_length, channels, height, width)
    """
    state_tensor = torch.tensor(world.get_state(), dtype=torch.float32, device=device)
    state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0) 
    s_h = state_tensor.repeat(population_size, 1, 1, 1, 1) 
    s_h = s_h.unsqueeze(2).repeat(1, 1, model.max_demo_length, 1, 1, 1) 

    return s_h

def rho_similarity(traj1, traj2):
    l = min(len(traj1), len(traj2))
    L = max(len(traj1), len(traj2))
    numer = 0
    for t in range(l):
        if traj1[t] == traj2[t]:
            numer += 1
        else:
            break
    return numer / L

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
    return max(alist)


np.set_printoptions(threshold=np.inf)

num_maps = 8
population_size = 100
Config.model_hidden_size = 256
# Config.data_batch_size = 1
# Config.data_num_demo_per_program = 1

tasks = ["StairClimber", "Maze", "FourCorners", 
         "Harvester", "CleanHouse", "TopOff",
         "DoorKey", "FindMarker", "Seeder", "OneStroke", "Snake"]

world_gen = WorldGenerator()

random_maps = [world_gen.generate(wall_prob=0.1, marker_prob=0.1) for _ in range(num_maps)]

dsl = DSL.init_default_karel()
device = torch.device('cpu')
model = load_model("LeapsVAE", dsl, device)
# task_cls = get_task_cls(Config.env_task)
params = torch.load("output/semantic_only_256_v2/model/best_val.ptp", map_location=device)
model.load_state_dict(params, strict=False)
model.eval()

latent_dim = model.hidden_size
latents = torch.randn(population_size, latent_dim, device=device)

s_h_list = [prepare_s_h(world) for world in random_maps]

a_h = torch.full(
    (1, 1, model.max_demo_length),
    fill_value=model.num_agent_actions - 1,
    dtype=torch.long,
    device=model.device
)
a_h_masks = torch.zeros_like(a_h, dtype=torch.bool, device=model.device)

with torch.no_grad():
    baseline_actions_list = []
    for s_h in s_h_list:
        pred_actions, _, _ = model.policy_executor(
            latents, s_h, a_h, a_h_masks, a_h_teacher_enforcing=False
        )
        baseline_actions_list.append(pred_actions.cpu().numpy())

num_iters = 10
sigma = 0.5

behavior_metrics = []
identity_metrics = []
current_latents = latents.clone()

for iter in range(num_iters):

    noise = sigma * torch.randn_like(current_latents, device=model.device)
    new_latents = current_latents + noise

    new_actions_list = []
    for s_h in s_h_list:
        with torch.no_grad():
            new_pred_actions, _, _ = model.policy_executor(
                new_latents, s_h, a_h, a_h_masks, a_h_teacher_enforcing=False
            )
        new_actions_list.append(new_pred_actions.cpu().numpy())

    print('baseline action:', baseline_actions_list)
    print('new actions', new_actions_list)
    
    seed_similarities = [] 
    seed_identities = [] 
    for i in range(population_size):
        sim_values = []
        id_values = []
        for world_idx in range(len(s_h_list)):
            baseline_seq = baseline_actions_list[world_idx][i]
            new_seq = new_actions_list[world_idx][i]
            
            print('BEFORE Baseline Seq: ', baseline_seq)
            print('BEFORE New seq: ', new_seq)

            baseline_seq = baseline_seq[baseline_seq != 5]
            new_seq = new_seq[new_seq != 5]

            print('AFTER Baseline Seq: ', baseline_seq)
            print('AFTER New seq: ', new_seq)
            lcp = longest_common_prefix(baseline_seq, new_seq)
            print('lcp: ', lcp)
        
            denom = max(len(baseline_seq), len(new_seq))
            if denom > 0:
                normalized_sim = lcp / max(len(baseline_seq), len(new_seq))
            else:
                normalized_sim = 0
            print('Normalized lcp: ', normalized_sim)

            sim_values.append(normalized_sim)
            id_values.append(1 if np.array_equal(baseline_seq, new_seq) else 0)

        seed_similarities.append(np.mean(sim_values))
        seed_identities.append(np.mean(id_values))

    mean_sim = np.mean(seed_similarities)
    std_sim = np.std(seed_similarities, ddof=1)
    se_sim = std_sim / np.sqrt(population_size)
    ci_sim = 1.96 * se_sim

    mean_id = np.mean(seed_identities)
    std_id = np.std(seed_identities, ddof=1)
    se_id = std_id / np.sqrt(population_size)
    ci_id = 1.96 * se_id

    behavior_metrics.append((mean_sim, mean_sim - ci_sim, mean_sim + ci_sim))
    identity_metrics.append((mean_id, mean_id - ci_id, mean_id + ci_id))

    print(f"Iteration {iter+1}:")
    print(f"  Mean Behavior Similarity = {mean_sim:.3f} (95% CI: [{mean_sim - ci_sim:.3f}, {mean_sim + ci_sim:.3f}])")
    print(f"  Mean Identity Rate = {mean_id:.3f} (95% CI: [{mean_id - ci_id:.3f}, {mean_id + ci_id:.3f}])")
    
    current_latents = new_latents

print("\nSummary over iterations:")
for i in range(num_iters):
    sim_mean, sim_lower, sim_upper = behavior_metrics[i]
    id_mean, id_lower, id_upper = identity_metrics[i]
    print(f"Iteration {i+1}:")
    print(f"  Behavior Similarity: {sim_mean:.3f} (95% CI: [{sim_lower:.3f}, {sim_upper:.3f}])")
    print(f"  Identity Rate: {id_mean:.3f} (95% CI: [{id_lower:.3f}, {id_upper:.3f}])")
