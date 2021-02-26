import numpy as np
from code.hawkes_process import exp_thinning_hawkes
from code.metric_estimator_classes import loglikelihood_estimator

if __name__ == "__main__":
    
    # Set seed
    np.random.seed(7)
    
    lambda_0 = 1.2
    alpha = -0.4
    beta = 0.9
    
    # Create timestamps from exponential Hawkes process
    hawkes = exp_thinning_hawkes(lambda_0, alpha, beta, max_jumps=100)
    hawkes.simulate()
    tList = hawkes.timestamps
    
    # Estimate using the estimator class and the real loglikelihood
    model = loglikelihood_estimator()
    model.fit(tList)
    
    print("Estimated parameters are:", model.estimator)         # Estimated parameters are: [ 1.49350942 -0.35850036  0.43912157]
    print("With a loglikelihood of:", -model.estimated_loss)    # With a loglikelihood of: -112.56222084702222  
