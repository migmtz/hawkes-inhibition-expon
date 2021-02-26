from hawkes_process import exp_thinning_hawkes
from metric_estimator_classes import loglikelihood_estimator

if __name__ == "__main__":

    np.random.seed(0)
    
    lambda_0 = 1.2
    alpha = -0.4
    beta = 0.9

    hawkes = exp_thinning_hawkes(lambda_0, alpha, beta, max_jumps=100)
    hawkes.simulate()
    tList = hawkes.timestamps
    
    model = loglikelihood_estimator()
    model.fit(tList)
    
    print(model.estimator)
    print(model.estimated_loss)
