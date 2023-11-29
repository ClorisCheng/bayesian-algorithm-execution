#%%
import os
import numpy as np
import matplotlib.pyplot as plt

import os
print(f"Current working directory: {os.getcwd()}")

# post_dir = "examples/posterior_sampling/results"
# bax_dir = "examples/posterior_sampling/results_bax"
post_dir = "./results"
bax_dir = "./results_bax"
save = True

post_regret_dir = os.path.join(post_dir, "regrets")
post_best_y_dir = os.path.join(post_dir, "best_y")
bax_regret_dir = os.path.join(bax_dir, "regrets")
bax_best_y_dir = os.path.join(bax_dir, "best_y")

post_best_y_arr = []
post_regret_arr = []
bax_best_y_arr = []
bax_regret_arr = []

for f in os.listdir(post_best_y_dir):
    # load npy files
    arr = np.load(os.path.join(post_best_y_dir, f))
    # print(f"arr.shape = {arr.shape}, f = {f}")
    
    post_best_y_arr.append(np.load(os.path.join(post_best_y_dir, f)))

iters = len(post_best_y_arr[0])
post_best_y_arr = np.vstack(post_best_y_arr)

for f in os.listdir(post_regret_dir):
    # load npy files
    post_regret_arr.append(np.load(os.path.join(post_regret_dir, f)))

post_regret_arr = np.vstack(post_regret_arr)
post_regret_arr = np.cumsum(post_regret_arr, axis=1)

for f in os.listdir(bax_best_y_dir):
    # load npy files
    bax_best_y_arr.append(np.load(os.path.join(bax_best_y_dir, f)))

bax_best_y_arr = np.vstack(bax_best_y_arr)

for f in os.listdir(bax_regret_dir):
    # load npy files
    bax_regret_arr.append(np.load(os.path.join(bax_regret_dir, f)))

bax_regret_arr = np.vstack(bax_regret_arr)
# find cumulative regret 
bax_regret_arr = np.cumsum(bax_regret_arr, axis=1)

#%%

# Plot Regret
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(np.arange(iters), np.mean(post_regret_arr, 0), label="Posterior Sampling")
ax.fill_between(np.arange(iters), 
                np.mean(post_regret_arr, 0) - np.std(post_regret_arr, 0), 
                np.mean(post_regret_arr, 0) + np.std(post_regret_arr, 0), 
                alpha=0.3)
ax.plot(np.arange(iters), np.mean(bax_regret_arr, 0), label="BAX")
ax.fill_between(np.arange(iters), 
                np.mean(bax_regret_arr, 0) - np.std(bax_regret_arr, 0), 
                np.mean(bax_regret_arr, 0) + np.std(bax_regret_arr, 0), 
                alpha=0.3)

ax.set_xlabel("Iteration")
ax.set_ylabel("Regret")
ax.legend()
ax.set_title("Cum. Regret Comparison")
if save:
    plt.savefig("regret", bbox_inches='tight')
plt.show()

#%%
# Plot Best y
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(np.arange(iters), np.mean(post_best_y_arr, 0), label="Posterior Sampling")
ax.fill_between(np.arange(iters), 
                np.mean(post_best_y_arr, 0) - np.std(post_best_y_arr, 0), 
                np.mean(post_best_y_arr, 0) + np.std(post_best_y_arr, 0), 
                alpha=0.3)
ax.plot(np.arange(iters), np.mean(bax_best_y_arr, 0), label="BAX")
ax.fill_between(np.arange(iters), 
                np.mean(bax_best_y_arr, 0) - np.std(bax_best_y_arr, 0), 
                np.mean(bax_best_y_arr, 0) + np.std(bax_best_y_arr, 0), 
                alpha=0.3)

ax.set_xlabel("Iteration")
ax.set_ylabel("Best y")
ax.legend()
ax.set_title("Best y Comparison")
if save:
    plt.savefig("best_y", bbox_inches='tight')

plt.show()
# %%
# plot log regret

fig, ax = plt.subplots(figsize=(6, 6))

ax.plot(np.arange(iters), np.log(np.mean(post_regret_arr, 0)), label="Posterior Sampling")
ax.fill_between(np.arange(iters), 
                np.log(np.mean(post_regret_arr, 0)) - np.std(np.log(post_regret_arr), 0), 
                np.log(np.mean(post_regret_arr, 0)) + np.std(np.log(post_regret_arr), 0), 
                alpha=0.3)
ax.plot(np.arange(iters), np.log(np.mean(bax_regret_arr, 0)), label="BAX")
ax.fill_between(np.arange(iters), 
                np.log(np.mean(bax_regret_arr, 0)) - np.std(np.log(bax_regret_arr), 0), 
                np.log(np.mean(bax_regret_arr, 0)) + np.std(np.log(bax_regret_arr), 0), 
                alpha=0.3)

ax.set_xlabel("Iteration")
ax.set_ylabel("Log Regret")
ax.legend()
ax.set_title("Log Cum. Regret Comparison")
if save:
    plt.savefig("log_regret", bbox_inches='tight')

plt.show()

# %%
# plot each individual trial in best_y
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(post_best_y_arr.shape[0]):
    ax.plot(np.arange(iters), post_best_y_arr[i, :], alpha=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Best y")
ax.set_title("Posterior Sampling (Post)")
if save:
    plt.savefig("post_best_y", bbox_inches='tight')
plt.show()
#%%
# plot each individual trial in best_y
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(bax_best_y_arr.shape[0]):
    ax.plot(np.arange(iters), bax_best_y_arr[i, :], alpha=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Best y")
ax.set_title("BAX")
if save:
    plt.savefig("post_best_y", bbox_inches='tight')
plt.show()
# %%
