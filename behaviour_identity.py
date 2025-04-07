import torch
import numpy as np
from vae.models import load_model
from dsl import DSL
from config import Config
from tasks import get_task_cls
from karel.world_generator import WorldGenerator


# Example usage
def prepare_s_h(world):
    """
    Converts a world into an s_h tensor of shape:
      (population_size, demos_per_program, max_demo_length, channels, height, width)
    """
    # world.get_state() returns a boolean NumPy array of shape (H, W, 16)
    # state_np = world.get_state()
    # Convert to float tensor and permute to (channels, H, W)
    state_tensor = torch.tensor(world.get_state(), dtype=torch.float32, device=device)
    state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]
    s_h = state_tensor.repeat(population_size, 1, 1, 1, 1) # [1, 1, C, H, W]
    s_h = s_h.unsqueeze(2).repeat(1, 1, model.max_demo_length, 1, 1, 1) # [population_size, 1, max_demo_length, C, H, W]

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
    """Compute the length of the longest common prefix between two 1D sequences."""
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


np.set_printoptions(threshold=np.inf)

num_maps = 8
population_size = 100
Config.model_hidden_size = 64
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
params = torch.load("output/semantic_only_64_mar/model/best_val.ptp", map_location=device)
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
    # For each world, get the predicted actions for each latent vector.
    baseline_actions_list = []
    for s_h in s_h_list:
        pred_actions, _, _ = model.policy_executor(
            latents, s_h, a_h, a_h_masks, a_h_teacher_enforcing=False
        )
        # pred_actions has shape (population_size, max_demo_length)
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
    
    # For each seed (latent vector), compute its behavior similarity and identity rate averaged across worlds.
    seed_similarities = []  # length = population_size
    seed_identities = []    # length = population_size
    for i in range(population_size):
        sim_values = []
        id_values = []
        # For each world, compare the new actions to the baseline actions for this seed.
        for world_idx in range(len(s_h_list)):
            baseline_seq = baseline_actions_list[world_idx][i]  # 1D array of length max_demo_length
            new_seq = new_actions_list[world_idx][i]
            lcp = longest_common_prefix(baseline_seq, new_seq)
            # Normalize by the maximum sequence length (typically model.max_demo_length)
            normalized_sim = lcp / model.max_demo_length
            sim_values.append(normalized_sim)
            id_values.append(1 if np.array_equal(baseline_seq, new_seq) else 0)
        # Average across worlds for this seed
        seed_similarities.append(np.mean(sim_values))
        seed_identities.append(np.mean(id_values))

    # Compute metrics across all worlds and all latent vectors
    # similarities_world = []  # will hold list of arrays (one per world)
    # identities_world = []    # same for identity rate
    # for world_idx in range(len(s_h_list)):
    #     similarities = []  # for each latent in this world
    #     identities = []
    #     baseline_actions = baseline_actions_list[world_idx]  # shape: (population_size, max_demo_length)
    #     new_actions = new_actions_list[world_idx]
    #     for i in range(population_size):
    #         seq_baseline = baseline_actions[i]
    #         seq_new = new_actions[i]
    #         # similarity_score = rho_similarity(seq_old, seq_new)
    #         lcp_length = longest_common_prefix(seq_baseline, seq_new)
    #         normalized_sim = lcp_length / max(len(seq_baseline), len(seq_new))
    #         similarities.append(normalized_sim)
    #         identities.append(1 if np.array_equal(seq_baseline, seq_new) else 0)
    #     similarities_world.append(np.mean(similarities))
    #     identities_world.append(np.mean(identities))
    
    # print('Similarities', similarities_world)
    # print('identities', identities_world)
    # exit(0)
    
    # Average metrics across worlds
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
    
    # mean_identity = np.mean(seed_identities)
    # behavior_similarities.append(mean_similarity)
    # identity_rates.append(mean_identity)

    print(f"Iteration {iter+1}:")
    print(f"  Mean Behavior Similarity = {mean_sim:.3f} (95% CI: [{mean_sim - ci_sim:.3f}, {mean_sim + ci_sim:.3f}])")
    print(f"  Mean Identity Rate = {mean_id:.3f} (95% CI: [{mean_id - ci_id:.3f}, {mean_id + ci_id:.3f}])")
    
    # print(f"Iteration {iter+1}: Mean Behavior Similarity = {mean_similarity:.3f}, Identity Rate = {mean_identity:.3f}")
    
    # Update for next iteration: use new latents and new actions (per world)
    current_latents = new_latents
    # current_actions_list = new_actions_list

# ----- Final Summary -----
print("\nSummary over iterations:")
for i in range(num_iters):
    sim_mean, sim_lower, sim_upper = behavior_metrics[i]
    id_mean, id_lower, id_upper = identity_metrics[i]
    print(f"Iteration {i+1}:")
    print(f"  Behavior Similarity: {sim_mean:.3f} (95% CI: [{sim_lower:.3f}, {sim_upper:.3f}])")
    print(f"  Identity Rate: {id_mean:.3f} (95% CI: [{id_lower:.3f}, {id_upper:.3f}])")
# print("\nSummary over iterations:")
# for i in range(num_iters):
#     print(f"Iteration {i+1}: Behavior Similarity = {behavior_similarities[i]:.3f}, Identity Rate = {identity_rates[i]:.3f}")


# for idx, world in enumerate(random_maps):
#     state_tensor = torch.tensor(world.get_state(), dtype = torch.float32, device = model.device)
#     state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]

#     s_h = state_tensor.repeat(1, 1, 1, 1, 1)  # [population_size, 1, C, H, W]
#     s_h = s_h.unsqueeze(2).repeat(1, 1, model.max_demo_length, 1, 1, 1)  # [population_size, 1, max_demo_length, C, H, W]

#     pred_actions, _, _ = model.policy_executor(
#     population,  # [population_size, hidden_size]
#     s_h=s_h,
#     a_h=a_h,
#     a_h_mask=a_h_masks,
#     a_h_teacher_enforcing=False
# )

#     print(f"Map {idx + 1}:\n{world.to_string()}\n{'-' * 30}\n")

# dsl = DSL.init_default_karel()
# device = torch.device('cpu')
# model = load_model("LeapsVAE", dsl, device)
# # task_cls = get_task_cls(Config.env_task)
# params = torch.load("output/semantic_only_64_mar/model/best_val.ptp", map_location=device)
# model.load_state_dict(params, strict=False)

# dummy_hidden = torch.randn(1, model.hidden_size, device=device)
# z = model.sample_latent_vector(dummy_hidden)

# world = WorldGenerator()
# world_state = torch.tensor(world.s.astype(np.float32)).permute(2, 0, 1) 

# pred_a_h, pred_a_h_logits, pred_a_h_masks = model.policy_executor(
#     z, s_h, a_h, a_h_mask, a_h_teacher_enforcing=False
# )

# print("Predicted actions:", pred_a_h)
# evaluate_space(model, task_cls, sigma=0.25)