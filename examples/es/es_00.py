import copy
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from bax.models.simple_gp import SimpleGp
from bax.alg.evolution_strategies import EvolutionStrategies
from bax.acq.acqoptimize import AcqOptimizer

import neatplot
neatplot.set_style('fonts')


seed = 11
np.random.seed(seed)

# Set function
xm = 0.3
xa = 4.0
ym = 2.0
#f = lambda x: ym * np.sin(np.pi * xm * (x[0] + xa)) + \
              #ym * np.sin(2 * xm * np.pi * (x[0] + xa)) / 2.0

# In this example, we will be maximizing f
f = lambda x: -(0.1 * x[0] - 1)**2 + 2

# Set data for model
data = Namespace()
data.x = [[0.0]]
data.y = [f(x) for x in data.x]

# Set model as a GP
gp_params = {'ls': 3.0, 'alpha': 2.0, 'sigma': 1e-2}
model = SimpleGp(gp_params)
model.set_data(data)

# Set arrays
min_x = 3.5
max_x = 20.0
x_test = [[x] for x in np.linspace(0.0, max_x, 500)]
y_test = [f(x) for x in x_test]

# Set algorithm
algo = EvolutionStrategies(
    {'n_generation': 10, 'n_population': 5, 'samp_str': 'mut'}
) # Code currently requires init to 0

# -----
#algo.print_str()

#from bax.models.function import FunctionSample
#fs = FunctionSample()
#fs.set_model(model)

#e, o = algo.run_algorithm_on_f(f)
# -----

# Pre-computed algorithm output on f:
algo_output_f = 8.597

n_iter = 40

for i in range(n_iter):
    # Plot setup
    fig = plt.figure(figsize=(8, 5))
    plt.xlim([0, max_x])
    plt.ylim([-7.0, 8.0])
    plt.xlabel('x')
    plt.ylabel('y')

    # Optimize acquisition function
    acqopt = AcqOptimizer({'x_test': x_test, 'acq_str': 'out'})
    x_next = acqopt.optimize(model, algo)

    # Update algo.init_x
    algo.params.init_x = data.x[np.argmax(data.y)]

    # Compute current expected output
    expected_output = np.mean(acqopt.get_last_output_list())
    out_abs_err = np.abs(algo_output_f - expected_output)

    # Print
    print(f'Acq optimizer x_next = {x_next}')
    print(f'Current expected_output = {expected_output}')
    print(f'Current output abs. error = {out_abs_err}')
    print(f'Finished iter i = {i}')

    # Plot true function
    plt.plot(x_test, y_test, '-', color='k', linewidth=2)

    # Show plot
    #neatplot.save_figure(f'gp_test_11_{i}', ['png'])
    plt.show()

    inp = input('Press enter to continue (any other key to stop): ')
    if inp:
        break
    plt.close()

    # Query function, update data
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)

    # Update model
    model = SimpleGp(gp_params)
    model.set_data(data)
