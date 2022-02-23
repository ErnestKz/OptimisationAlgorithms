import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['figure.facecolor'] = '1'
import matplotlib.pyplot as plt

from sympy import diff, lambdify, symbols, init_printing, Max, Abs
init_printing(use_unicode=False)

def adagrad(dfs,
            x0,   # this is a vector, the input to f/df
            i_max=50):
    
    x_history = [x0]                             # keep track of history of xi's
    for _ in range(i_max):
        subgradient_step_vector = []             # each partial gets its own step

        
        for df in dfs:
            subgradient_step = adagrad_partial_alpha(df, x_history, alpha0=0.1, eps=0.001) * df(x[-1])
            subgradient_step_vector += [ subgradient_step ]
        x = x - subgradient_step_vector; x_history += [x]
    return x

def finite_diff(f, x, delta=0.01):
    return (f(x) - f(x - delta)) / delta
