import numpy as np
from scipy import stats
from scipy.optimize import minimize
from code.hawkes_process import exp_thinning_hawkes
from code.likelihood_functions import *


class loglikelihood_estimator(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss.
    Contemplated losses are functions from likelihood_functions, either loglikelihood or likelihood_approximated, or a callable.

    Attributes
    ----------
    estimator : array
        Array containing estimated parameters.
    estimated_loss : float
        Value of loss at estimated parameters.
    model : object "exp_thinning_hawkes"
        Class containing the estimated parameters, timestamps and corresponding intensities. Exists only if return_model is set to True.
    """
    def __init__(self, loss=loglikelihood, solver="L-BFGS-B", C=None, initial_guess=np.array((1.0, 0.0, 1.0)), simplex=True, bounds=[(0.0, None), (None, None), (0.0, None)], return_model=False, options = {'disp': False}):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated}
            Function to minimize. Default is loglikelihood.
        solver : {"L-BFGS-B", "nelder-mead"}
            Solver used in function minimize from scipy.optimize. 
            "L-BFGS-B" uses the bounds argument and "nelder-mead" the simplex argument.
            Default is "L-BFGS-B".
        C : float
            Penalty constant. Only taken in account if not None. Default is None.
        initial_guess : array of float.
            Initial guess for estimated vector. When using "nelder-mead", it is only used if simplex is False. Default is np.array((1.0, 0.0, 1.0)).
        simplex : bool
            Whether if initialize the solver with a simplex. 
            The simplex is then initialized randomly in four points where the loglikelihood is finite. Default is True.
        bounds : list.
            Bounds to set for the algorithm. By default, only bounds are for lambda_0 and beta to be non-negative.
            If method appears to be unstable, a bounds like ((epsilon, None), (None, None), (epsilon, None)) with epsilon = 1e-10 is recommended.
            Default is ((0.0, None), (None, None), (0.0, None)).
        return_model : bool
            Whether to create an object "exp_thinning_hawkes" with obtained estimation. 
            The class has its corresponding intensity function.
            
        """
        if C is not None:
            self.loss = lambda theta, timestamps: loss(theta, timestamps) + C*(theta[0]**2 + theta[1]**2 + theta[2]**2)
            self.C = C
        else:
            self.loss = loss
        self.solver = solver
        self.initial_guess = initial_guess
        self.simplex = simplex
        self.bounds = bounds
        self.return_model = return_model
        self.options = options
        
        if solver == "L-BFGS-B":
            self._estimator = loglikelihood_estimator_bfgs(loss=self.loss, bounds=self.bounds, initial_guess=self.initial_guess, options=self.options)
        elif solver == "nelder-mead":
            self._estimator = loglikelihood_estimator_nelder(loss=self.loss, simplex=self.simplex, initial_guess=self.initial_guess, options=self.options)
        else:
            raise ValueError('Unknown solver %s' % solver)

    def fit(self, timestamps):
        """
        Parameters
        ----------
        timestamps : list of float
            Ordered list containing event times.
        """

        self.res = self._estimator.fit(timestamps)

        self.estimator = self.res.x
        self.estimated_loss = self.res.fun

        if self.return_model:
            self.model = exp_thinning_hawkes(self.estimator[0], self.estimator[1], self.estimator[2])
            self.model.set_time_intensity(timestamps)



class loglikelihood_estimator_bfgs(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the L-BFGS-B algorithm.

    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """
    def __init__(self, loss=loglikelihood, bounds=[(0.0, None), (None, None), (0.0, None)], initial_guess=np.array((1.0, 0.0, 1.0)), options = {'disp': False}):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
        bounds : list.
            Bounds to set for the algorithm. By default, only bounds are for lambda_0 and beta to be non-negative.
            If method appears to be unstable, a bounds like ((epsilon, None), (None, None), (epsilon, None)) with epsilon = 1e-10 is recommended.
            Default is ((0.0, None), (None, None), (0.0, None)).
        initial_guess : array of float.
            Initial guess for estimated vector. Default is np.array((1.0, 0.0, 1.0)).
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.
        """
        self.loss = loss
        self.bounds = bounds
        self.initial_guess = np.array((1.0, 0.0, 1.0))
        self.options = options

    def fit(self, timestamps):
        """
        Parameters
        ----------
        timestamps : list of float
            Ordered list containing event times.
        """

        self.res = minimize(self.loss, self.initial_guess, method="L-BFGS-B",
                       args=timestamps, bounds=self.bounds,
                       options=self.options)

        return(self.res)


class loglikelihood_estimator_nelder(object):
    """
    Estimator class for Exponential Hawkes process obtained through minimizaton of a loss using the Nelder-Mead simplex algorithm.
    
    Attributes
    ----------
    res : OptimizeResult
        Result from minimization.
    """
    def __init__(self, loss=loglikelihood, simplex=True, initial_guess=np.array((1.0, 0.0, 1.0)), options = {'disp': False}):
        """
        Parameters
        ----------
        loss : {loglikelihood, likelihood_approximated} or callable.
            Function to minimize. Default is loglikelihood.
        simplex : bool
            Whether if initialize the solver with a simplex. 
            The simplex is then initialized randomly in four points where the loglikelihood is finite. Default is True.
        initial_guess : array of float.
            Initial guess for estimated vector. Used only if simples is False. Default is np.array((1.0, 0.0, 1.0)).
        options : dict
            Options to pass to the minimization method. Default is {'disp': False}.
        """
        self.loss = loss
        self.simplex = simplex
        self.initial_guess = initial_guess
        self.options = options

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

        self.res = minimize(self.loss, self.initial_guess, method="nelder-mead",
                       args=timestamps, options=self.options)

        return(self.res)

