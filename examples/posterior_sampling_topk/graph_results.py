#%%
import os
import numpy as np
import matplotlib.pyplot as plt

import os
print(f"Current working directory: {os.getcwd()}")

# post_dir = "examples/posterior_sampling_topk/results"
# bax_dir = "examples/posterior_sampling_topk/results_bax"


post_dir = "./results"

save = True

metrics = ["jacc", "norm"]

strategy_lst = ["var", "bax", "rdm"]
jacc_arrs = {}
norm_arrs = {}

for strategy in strategy_lst:
    for f in os.listdir(post_dir):
        if f.endswith(".npy"):
            if "jacc" in f:
                l = jacc_arrs.get(strategy, [])
                l.append(np.load(os.path.join(post_dir, f)))
                jacc_arrs[strategy] = l
            elif "norm" in f:
                l = norm_arrs.get(strategy, [])
                l.append(np.load(os.path.join(post_dir, f)))
                norm_arrs[strategy] = l
#%%

bax_dir = "./results_bax"
for f in os.listdir(bax_dir):
    if f.endswith(".npy"):
        if "jacc" in f:
            l = jacc_arrs.get("bax", [])
            l.append(np.load(os.path.join(bax_dir, f)))
            jacc_arrs["bax"] = l
        elif "norm" in f:
            l = norm_arrs.get("bax", [])
            l.append(np.load(os.path.join(bax_dir, f)))
            norm_arrs["bax"] = l

#%%
# Plot jacc
fig, ax = plt.subplots(figsize=(6, 6))
for strategy, arrs in jacc_arrs.items():
    arr = np.array(arrs)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    ax.plot(mean, label=strategy)
    ax.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Jaccard Similarity")
ax.set_title("Jaccard Similarity vs Iteration")
ax.legend()
if save:
    plt.savefig("jaccard.png", dpi=300)

#%%
# Plot norm
fig, ax = plt.subplots(figsize=(6, 6))
for strategy, arrs in norm_arrs.items():
    arr = np.array(arrs)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    ax.plot(mean, label=strategy)
    ax.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Norm")
ax.set_title("Norm vs Iteration")
ax.legend()
if save:
    plt.savefig("norm.png", dpi=300)
#%%
