# Algorithms.py

# Algorithms implement a similar inteface:
# - specific names on input arguments
# - accesses function related things through the OptimisableFunction class
# - needs to return X, Y

import numpy as np

standard_parameters = ("x0", "f", "iters")

class OptimisationAlgorithm:
    def __init__(self, algorithm, algorithm_name):
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name
        
        arguments = algorithm.__code__.co_varnames[:algorithm.__code__.co_argcount]
        self.hyperparameters = list(filter(lambda arg: arg not in standard_parameters, arguments))

def polyak(x0, f, f_star, eps, iters):
    dfs = f.partial_derivatives ; f = f.function ; x = x0 ; X = [x] ; Y = [f(*x)]
    
    for _ in range(iters):
        fdif = f(*x) - f_star
        df_squared_sum = np.sum(np.array([df(*x)**2 for df in dfs]))
        alpha = fdif / (df_squared_sum + eps)
        x = x - alpha * np.array([df(*x) for df in dfs])

        X += [x] ; Y += [f(*x)]
    return X, Y

Polyak = OptimisationAlgorithm(algorithm=polyak,
                               algorithm_name="Polyak")

def rmsprop(x0, f, alpha0, beta, eps, iters):
    dfs = f.partial_derivatives ; f = f.function ; x = x0 ; X = [x] ; Y = [f(*x)]
    
    sum = np.zeros(len(dfs)) ; alpha = alpha0
    for _ in range(iters):
      x = x - (alpha * np.array([df(*x) for df in dfs]))
      sum = beta * sum + (1 - beta) * np.array([df(*x)**2 for df in dfs]) 
      alpha = alpha0 / (np.sqrt(sum) + eps)
      
      X += [x] ; Y += [f(*x)]
    return X, Y

RMSProp = OptimisationAlgorithm(algorithm=rmsprop,
                                algorithm_name="RMSProp")


def heavy_ball(x0, f, alpha, beta, iters):
    dfs = f.partial_derivatives ; f = f.function ; x = x0 ; X = [x] ; Y = [f(*x)]
    
    z = np.zeros(len(dfs))
    for _ in range(iters):
        z = beta * z + alpha * np.array([df(*x) for df in dfs])
        x = x - z

        X += [x] ; Y += [f(*x)]
    return X, Y

HeavyBall = OptimisationAlgorithm(algorithm=heavy_ball,
                                  algorithm_name="Heavy Ball")

def adam(x0, f, eps, beta1, beta2, alpha, iters):
    dfs = f.partial_derivatives ; f = f.function ; x = x0 ; X = [x] ; Y = [f(*x)]
    
    m = np.zeros(len(dfs)) ; np.zeros(len(dfs))
    for k in range(iters_max):
        i = k + 1
        m = beta1 * m + (1 - beta1) * np.array([df(*x) for df in dfs])
        v = beta2 * v + (1 - beta2) * np.array([(df(*x)**2) for df in dfs])
        mhat = (m / (1 - beta1**i)) 
        vhat = (v / (1 - beta2**i))
        x = x - alpha * (mhat / (np.sqrt(vhat) + eps))
        
        X += [x] ; Y += [f(*x)]
    return X,Y

Adam = OptimisationAlgorithm(algorithm=adam,
                             algorithm_name="Adam")
