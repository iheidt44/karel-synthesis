import matplotlib.pyplot as plt
import pandas as pd

# sigmas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25]
sigmas = [1.5]
models = ["NewLatent", "LEAPS"]
num_iters = 10

# sigma_colors = {
#     0.1: 'red',
#     0.25: 'red',
#     0.5: 'red',
#     0.75: 'red',
#     1.0: 'blue',
#     1.25: 'green'
# }

model_colors = {
    "NewLatent": 'red',
    "LEAPS": 'blue'
}

results = {model: {sigma: {"behavior": [], "identity": []} for sigma in sigmas} for model in models}

for model in models:
    for sigma in sigmas:
        behavior_file = f"behaviour_data/behaviour_{model}_{sigma}.txt"
        behavior_df = pd.read_csv(behavior_file)
        results[model][sigma]["behavior"] = list(zip(
            behavior_df["Mean"], 
            behavior_df["Lower_CI"], 
            behavior_df["Upper_CI"]
        ))
        
        identity_file = f"behaviour_data/identity_{model}_{sigma}.txt"
        identity_df = pd.read_csv(identity_file)
        results[model][sigma]["identity"] = list(zip(
            identity_df["Mean"], 
            identity_df["Lower_CI"], 
            identity_df["Upper_CI"]
        ))

for sigma in sigmas:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for model in models:
        behavior = results[model][sigma]["behavior"]
        mean = [m[0] for m in behavior]
        lower = [m[1] for m in behavior]
        upper = [m[2] for m in behavior]
        x = range(1, num_iters + 1)
        label = f"{model} σ={sigma}"
        color = model_colors[model]
        linestyle = '--' if model == "NewLatent" else '-'
        plt.plot(x, mean, label=label, color=color, linestyle=linestyle)
        plt.fill_between(x, lower, upper, alpha=0.2, color=color)
    plt.xlabel("Number of Mutations")
    plt.ylabel("Behavior Similarity")
    plt.legend()
    plt.title(f"Behavior Similarity (σ={sigma})")
    plt.ylim(0, 1)
    plt.xlim(1, 10)
    
    plt.subplot(1, 2, 2)
    for model in models:
        identity = results[model][sigma]["identity"]
        mean = [m[0] for m in identity]
        lower = [m[1] for m in identity]
        upper = [m[2] for m in identity]
        x = range(1, num_iters + 1)
        label = f"{model} σ={sigma}"
        color = model_colors[model]
        linestyle = '--' if model == "NewLatent" else '-'
        plt.plot(x, mean, label=label, color=color, linestyle=linestyle)
        plt.fill_between(x, lower, upper, alpha=0.2, color=color)
    plt.xlabel("Number of Mutations")
    plt.ylabel("Identity Rate")
    plt.legend()
    plt.title(f"Identity Rate (σ={sigma})")
    plt.ylim(0, 1)
    plt.xlim(1, 10)
    
    plt.tight_layout()
    plt.savefig(f"behaviour_identity_{sigma}.png")
    plt.show() 
