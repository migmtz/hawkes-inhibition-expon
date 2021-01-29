import numpy as np
from matplotlib import pyplot as plt
from hawkes_process import exp_thinning_hawkes

# Fix the example's random seed.
np.random.seed(0)
# Create a process with given parameters and maximal number of jumps.
hawkes = exp_thinning_hawkes(lambda_0=1.05, alpha=-0.7, beta=0.8, max_jumps=15)
hawkes.simulate()
# Plotting function of intensity and step functions.
hawkes.plot_intensity()

plt.show()
