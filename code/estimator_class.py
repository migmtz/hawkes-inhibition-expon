import numpy as np
from scipy import stats
from scipy.optimize import minimize
from hawkes_process import exp_thinning_hawkes
from likelihood_functions import *


class loglikelihood_estimator(object):
        """
        Estimator class for Exponential Hawkes process obtained through minimizaton of a loss.
        Contemplated losses are functions from likelihood_functions, either loglikelihood or likelihood_approximated.

        Attributes
        ----------
        estimator : array
            Array containing estimated parameters.
        estimated_loss : float
            Value of loss at estimated parameters.
        model : object "exp_thinning_hawkes"
            Class containing the estimated parameters, timestamps and corresponding intensities. Exists only if return_model is set to True.
        """
    def __init__(self, loss=loglikelihood, solver="nelder-mead", simplex=True,penalty=False, C=1.0, return_model=False):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated}
            Function to minimize. Default is loglikelihood.
        solver : {"nelder-mead"}
            Solver used in function minimize from scipy.optimize. 
            Only contemplated case is "nelder-mead" which allows to obtain the better results along with the simplex initialization
        simplex : bool
            Whether if initialize the solver with a simplex. Default is True.
        penalty : bool
            Whether to add an L2-penalty. Default is False.
        C : float
            Penalty constant. Default is 1.0.
        return_model : bool
            Whether to create an object "exp_thinning_hawkes" with obtained estimation. 
            The class has its corresponding intensity function.
        """
        if penalty:
            self.loss = lambda theta, timestamps: loss(theta, timestamps) + C*(theta[0]**2 + theta[1]**2 + theta[2]**2)
        else:
            self.loss = loss
        self.options = {'xatol': 1e-8, 'disp': False}
        self.solver = solver
        self.simplex = simplex
        self.penalty = penalty
        self.C = C
        self.return_model = return_model

    def fit(self, timestamps):
        """
        Parameters
        ----------
        timestamps : list of float
            Ordered list containing event times.
        """
        if self.simplex:
            x_simplex = []

            while len(x_simplex) != 4:
                candidate = np.array([np.random.normal(0, 1), np.random.normal(0, 1), np.random.normal(0, 1)])
                like = self.loss(candidate, timestamps)
                if like < 100000.0:
                    x_simplex += [candidate]

            self.options['initial_simplex'] = x_simplex

        self.res = minimize(self.loss, np.zeros(3), method=self.solver,
                       args=timestamps,
                       options=self.options)

        self.estimator = self.res.x
        self.estimated_loss = self.res.fun

        if self.return_model:
            self.model = exp_thinning_hawkes(self.estimator[0], self.estimator[1], self.estimator[2])
            self.model.set_time_intensity(timestamps)

