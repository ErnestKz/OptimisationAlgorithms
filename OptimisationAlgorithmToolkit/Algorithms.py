# Algorithms.py

# Algorithms implement a similar inteface:
# - specific names on input arguments
# - accesses function related things through the OptimisableFunction class
# - needs to return X, Y

import numpy as np

class OptimisationAlgorithm:
    def __init__(self, algorithm, algorithm_name):
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name
        
        arguments = algorithm.__code__.co_varnames[:algorithm.__code__.co_argcount]
        self.all_parameters = arguments
        self.standard_parameters = ("x0", "f", "iters")
        self.hyperparameters = list(filter(lambda arg: arg not in self.standard_parameters, arguments))

    def __type_check_parameters(self, input_record):
        for key in input_record.keys():
            if key not in self.all_parameters:
                raise NameError(key + " is not one of: " + str(self.all_parameters))
        for key in self.all_parameters:
            if key not in input_record:
                raise NameError(key + " is missing from input: " + str(list(input_record.keys())))
            
    def set_parameters(self, **input_record):
        self.__type_check_parameters(input_record)
        self.parameter_values = input_record
        return self

    def run(self):
        inputs = self.__make_input()
        for input in inputs:
            input["X"], input["Y"] = self.algorithm(**input)
            input["X"] = np.array(input["X"])
            input["Y"] = np.array(input["Y"])
            input["algorithm"] = self
        return inputs

    def __make_input(self):
        kwargs = self.parameter_values.copy()
        expected_vector = { "x0" }
        for key, value in kwargs.items():
            if key in expected_vector:
                value = np.array(value)
                if value.ndim == 1:
                    kwargs[key] = [value]
            else:
                if type(value) is not list:
                    kwargs[key] = [value]

        keys = kwargs.keys()
        partial_dicts = [{}]
        for key in keys:
            partial_dicts_new = []
            for partial_dict in partial_dicts:
                for value in kwargs[key]: # making a new partial dict for each value
                    partial_dict_new = partial_dict.copy()
                    partial_dict_new[key] = value
                    partial_dicts_new += [partial_dict_new]
                    partial_dicts = partial_dicts_new
        return partial_dicts

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

def constant_step(x0, alpha, f, iters):
    dfs = f.partial_derivatives ; f = f.function ; x = x0 ; X = [x] ; Y = [f(*x)]
    
    for _ in range(iters):
        step = alpha * np.array([df(*x) for df in dfs])
        x = x - step
        
        X += [x] ; Y += [f(*x)]
    return X, Y

ConstantStep = OptimisationAlgorithm(algorithm=constant_step,
                                algorithm_name="Constant")

def adagrad(x0, f, alpha0, eps, iters):
    dfs = f.partial_derivatives ; f = f.function ; x = x0 ; X = [x] ; Y = [f(*x)]
    
    df_vector_sum = np.zeros(len(dfs))
    for _ in range(iters):
        df_vec = np.array([df(*x) for df in dfs])
        df_vector_sum += df_vec**2
        alphas = alpha0 / (np.sqrt(df_vector_sum) + eps)
        x = x  - (alphas * df_vec)
        
        X += [x] ; Y += [f(*x)]
    return X, Y

Adagrad = OptimisationAlgorithm(algorithm=adagrad,
                                algorithm_name="Adagrad")

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
    
    m = np.zeros(len(dfs)) ; v = np.zeros(len(dfs))
    for k in range(iters):
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
