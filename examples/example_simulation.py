import numpy as np
from matplotlib import pyplot as plt
from code.hawkes_process import exp_thinning_hawkes

if __name__ == "__main__":

    # Set seed
    np.random.seed(0)

    lambda_0 = 1.2
    alpha = -0.4
    beta = 0.9
    
    # Create a process with given parameters and maximal number of jumps.
    hawkes = exp_thinning_hawkes(lambda_0=lambda_0, alpha=alpha, beta=beta, max_jumps=15)
    hawkes.simulate()
    
    # Plotting function of intensity and step functions.
    hawkes.plot_intensity()

    plt.show()
