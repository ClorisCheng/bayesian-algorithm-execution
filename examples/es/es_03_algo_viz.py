import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
#plt.ion()

from bax.models.simple_gp import SimpleGp
from bax.alg.evolution_strategies import EvolutionStrategies
from bax.acq.visualize import AcqViz2D

from branin import branin, unif_random_sample_domain

import neatplot
#neatplot.set_style('fonts')


seed = 11
np.random.seed(seed)

# Set function
f = branin
domain = [[-5, 10], [0, 15]]
init_x = [4.8, 13.0]
#init_x = [4.0, 14.0]
#init_x = [5.7, 13.25]
#init_x = [7.5, 13.0]

# Set data for model
data = Namespace()
data.x = [init_x]
data.y = [f(x) for x in data.x]

# Set model as a GP
gp_params = {'ls': 3.0, 'alpha': 2.0, 'sigma': 1e-2}
model = SimpleGp(gp_params)
model.set_data(data)

# Set algorithm
algo = EvolutionStrategies(
    {
        'n_generation': 20,
        'n_population': 8,
        'samp_str': 'mut',
        'init_x': init_x,
        'domain': domain,
    }
)

# Run algorithm on f
exe_path, output = algo.run_algorithm_on_f(f)

# Plot setup
fig = plt.figure(figsize=(6, 6))

plt.xlim([-5.1, 10.1])
plt.ylim([-0.1, 15.1])

plt.xlabel('x')
plt.ylabel('y')

im = plt.imread('examples/es/branin_contour.png')
implot = plt.imshow(
    im, extent=[domain[0][0], domain[0][1], domain[1][0], domain[1][1]]
)

# Visualize execution path
vizzer = AcqViz2D()
h1 = vizzer.plot_model_data(exe_path)
h2 = vizzer.plot_exe_path_samples([exe_path], [output])
h3 = vizzer.plot_expected_output([output])
vizzer.make_legend([h1[0], h2[0], h3[0]])

# Print
min_idx = np.argmin(exe_path.y)
print(f'Best point so far x*: {exe_path.x[min_idx]}')
print(f'Value of best point so far f(x*): {exe_path.y[min_idx]}')
print(f'Found at iter: {min_idx}')

# Show plot
plt.show()
