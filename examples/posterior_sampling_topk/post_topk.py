import copy
from argparse import Namespace, ArgumentParser
from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# print current working directory
import os
print(f"Current working directory: {os.getcwd()}")

# set path for importing modules
import sys
sys.path.append("./")
print(f"Current working directory: {os.getcwd()}")

from bax.alg.algorithms import TopK
from bax.models.simple_gp import SimpleGp
from bax.models.gpfs_gp import GpfsGp
from bax.models.stan_gp import get_stangp_hypers
from bax.acq.acquisition import (
    BaxAcqFunction, UsBaxAcqFunction, EigfBaxAcqFunction, RandBaxAcqFunction
)
from bax.acq.acqoptimize import AcqOptimizer
from bax.acq.postsample import PostSampler # NEW
from bax.acq.visualize import AcqViz1D
from bax.util.domain_util import unif_random_sample_domain

# import neatplot
# neatplot.set_style("fonts")
# neatplot.update_rc('font.size', 20)


# Parse args
parser = ArgumentParser()
# parser.add_argument("--seed", type=int, default=12)
parser.add_argument("--n_init", type=int, default=1)
parser.add_argument("--n_iter", type=int, default=100)
args = parser.parse_args()

# # Set seeds
# print(f"*[INFO] Seed: {args.seed}")
# np.random.seed(args.seed)
# tf.random.set_seed(args.seed)

# def run_algo_on_mean_f(model_mf, algo_mf, n_samp_mf):
#     """Run algorithm on posterior mean (via MC estimate with n_samp samples)."""
#     model_mf.initialize_function_sample_list(n_samp_mf)
#     f_list = model_mf.call_function_sample_list
#     f_mf = lambda x: np.mean(f_list([x for _ in range(n_samp_mf)]))
#     exe_path_mf, output_mf = algo_mf.run_algorithm_on_f(f_mf)
#     return exe_path_mf, output_mf

# =========================== #

save_figure = False
save_result = False
strategy = "rdm"
n_trials = 10
seed_lst = [2009, 1732, 5644, 9645, 8181, 8455, 1524,  285, 8840, 8544]

# =========================== #

for seed in seed_lst[:n_trials]:
    print(f"*[INFO] Seed: {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Set function
    f_0 = lambda x: 2 * np.abs(x) * np.sin(x)
    f = lambda x_list: np.sum([f_0(x) for x in x_list])

    # Set vectorized function (for contour plot)
    @np.vectorize
    def f_vec(x, y):
        """Return f on input = (x, y)."""
        return f((x, y))

    # Set algorithm  details
    n_dim = 2
    domain = [[-10, 10]] * n_dim
    len_path = 150
    k = 10
    x_path = unif_random_sample_domain(domain, len_path)
    algo = TopK({"x_path": x_path, "k": k})

    # Get ground truth algorithm output
    algo_gt = TopK({"x_path": x_path, "k": k, "name": "groundtruth"})
    exepath_gt, output_gt = algo_gt.run_algorithm_on_f(f)
    print(f"Algorithm ground truth output is:\n{output_gt}")

    # Set metric
    metric_jacc = lambda x: algo.output_dist_fn_jaccard(x, output_gt)
    metric_norm = lambda x: algo.output_dist_fn_norm(x, output_gt)

    # Set data for model
    data = Namespace()
    data.x = unif_random_sample_domain(domain, args.n_init)
    data.y = [f(x) for x in data.x]

    # Set model details
    gp_params = {"ls": 2.5, "alpha": 20.0, "sigma": 1e-2, "n_dimx": n_dim}
    modelclass = GpfsGp

    # Set acquisition details
    acqfn_params = {"acq_str": "exe", "n_path": 100, "crop": True}
    acq_cls = BaxAcqFunction

    # Set acqopt details
    n_acqopt = 1500


    if save_result:
        # Set up results directory
        # results_dir = Path("topk_results_rebuttal")
        results_dir = Path("examples/posterior_sampling_topk/results")
        results_dir.mkdir(parents=True, exist_ok=True)

    if save_figure:
        # Set up img directory
        img_dir = Path("examples/posterior_sampling_topk/figures")
        img_dir = img_dir /  f'{strategy}_{seed}'
        img_dir.mkdir(parents=True, exist_ok=True)

    # Namespace to save results
    results = Namespace(
        expected_metric_jacc_list = [], expected_metric_norm_list = []
    )

    # Run BAX loop
    for i in range(args.n_iter):
        # Set model
        model = modelclass(gp_params, data)

        # Set and optimize acquisition function
        acqfn = acq_cls(acqfn_params, model, algo)
        x_test = unif_random_sample_domain(domain, n=n_acqopt)
        # acqopt = AcqOptimizer({"x_batch": x_test})
        # x_next = acqopt.optimize(acqfn)
        postsampler = PostSampler({"x_batch": x_test})
        x_next_list = postsampler.optimize(acqfn, algo_name = "topk")
        post_mean, post_std = model.get_post_mu_cov(x_next_list, full_cov=False)

        if strategy == "var":
            x_next = x_next_list[np.argmax(post_std)]
        elif strategy == "rdm":
            x_next = x_next_list[np.random.randint(len(x_next_list))]
        elif strategy == "mean":
            x_next = x_next_list[np.argmax(post_mean)]


        # Get expected metric
        metric_jacc_list = [metric_jacc(output) for output in acqfn.output_list]
        metric_norm_list = [metric_norm(output) for output in acqfn.output_list]
        expected_metric_jacc = np.mean(metric_jacc_list)
        expected_metric_norm = np.mean(metric_norm_list)

        # Print iter info
        print(f"Acqopt x_next = {x_next}")
        print(f"output_list[0] = {acqfn.output_list[0]}")
        print(f"expected_metric_jacc = {expected_metric_jacc}")
        print(f"expected_metric_norm = {expected_metric_norm}")
        print(f"Finished iter i = {i}")

        # Update results namespace
        results.expected_metric_jacc_list.append(expected_metric_jacc)
        results.expected_metric_norm_list.append(expected_metric_norm)
        
        if save_figure:
        # Plot
            fig, ax = plt.subplots(figsize=(6, 6))
            # -- plot function contour
            grid = 0.1
            xpts = np.arange(domain[0][0], domain[0][1], grid)
            ypts = np.arange(domain[1][0], domain[1][1], grid)
            X, Y = np.meshgrid(xpts, ypts)
            Z = f_vec(X, Y)
            ax.contour(X, Y, Z, 20, cmap=cm.Greens_r, zorder=0)
            # -- plot top_k
            topk_arr = np.array(output_gt.x)
            ax.plot(
                topk_arr[:, 0],
                topk_arr[:, 1],
                '*',
                marker='*',
                markersize=30,
                color='gold',
                markeredgecolor='black',
                markeredgewidth=0.05,
                zorder=1
            )
            # -- plot x_path
            x_path_arr = np.array(x_path)
            ax.plot(x_path_arr[:, 0], x_path_arr[:, 1], '.', color='#C0C0C0', markersize=8)
            # -- plot observations
            for x in data.x:
                ax.scatter(x[0], x[1], color=(0, 0, 0, 1), s=80)
            # -- plot x_next
            ax.scatter(x_next[0], x_next[1], color='deeppink', s=80, zorder=10)
            # -- plot estimated output
            # for out in acqfn.output_list:
            #     out_arr = np.array(out.x)
            #     ax.plot(
            #         out_arr[:, 0], out_arr[:, 1], 's', markersize=8, color='blue', alpha=0.02
            #     )
            # -- plot x_next_list 
            x_next_list_arr = np.array(x_next_list)
            # make scatter plot with shape x
            ax.scatter(x_next_list_arr[:, 0], x_next_list_arr[:, 1], marker='x', color='cyan', s=50)

            # -- lims, labels, titles, etc
            ax.set(xlim=domain[0], ylim=domain[1])
            ax.set_title("Posterior Sampling with Top-k Algorithm")

            # Save plot
            img_path = img_dir / f'topk_{i}'
            print(f"Saving image to {img_path}")
            plt.savefig(str(img_path), bbox_inches='tight')
            # neatplot.save_figure(str(img_path), 'pdf')

        # Query function, update data
        y_next = f(x_next)
        data.x.append(x_next)
        data.y.append(y_next)


    if save_result:
        # Save final data
        results.data = data

        # # Pickle results
        # file_str = f"{strategy}_{args.seed}.pkl"
        # with open(results_dir / file_str, "wb") as handle:
        #     pickle.dump(results, handle)
        #     print(f"Saved results file: {results_dir}/{file_str}")
        
        np.save(f"{results_dir}/{strategy}_jacc_{seed}.npy", results.expected_metric_jacc_list)
        np.save(f"{results_dir}/{strategy}_norm_{seed}.npy", results.expected_metric_norm_list)
        print(f"Saved results file: {results_dir}")