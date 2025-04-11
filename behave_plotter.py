import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

models = ["NewLatent", "LEAPS"]
num_iters = 10

sigmas = [0.75, 1.0, 1.25]

sigma_colors = {
    0.75: 'red',
    1.0: 'blue',
    1.25: 'green'
}

results = {"NewLatent": {}, "LEAPS": {}}
for model_name in models:
    for sigma in sigmas:
        behavior_file = f"behaviour_{model_name}_{sigma}.txt"
        behavior_metrics = []

        with open(behavior_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                iteration, mean, lower, upper = map(float, row)
                behavior_metrics.append((mean, lower, upper))
        results[model_name][sigma] = {"behavior": behavior_metrics}

        identity_file = f"identity_{model_name}_{sigma}.txt"
        identity_metrics = []

        with open(identity_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                iteration, mean, lower, upper = map(float, row)
                identity_metrics.append((mean, lower, upper))
        results[model_name][sigma]["identity"] = identity_metrics

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
plt.savefig("figure4_latent_only1.png")
plt.show()