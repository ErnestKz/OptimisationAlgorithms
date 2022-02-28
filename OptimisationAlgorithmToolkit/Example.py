from Function import OptimisableFunction
from Algorithms import Polyak, Adam, HeavyBall, RMSProp
from Datatype import make_input, run_optimisations
from Plotting import plot_y_vary_param

from sympy import symbols, Max, Abs

x1, x2 = symbols('x1 x2', real=True)
sym_f1 = 3 * (x1-9)**4 + 5 * (x2-9)**2
f1 = OptimisableFunction(sym_f1, [x1, x2], "f_1")

sym_f2 = Max(x1-9 ,0) + 5 * Abs(x2-9)
f2 = OptimisableFunction(sym_f2, [x1, x2], "f_2")

algs = [Polyak, Adam, HeavyBall, RMSProp]
for a in algs:
    print(a.algorithm_name, a.hyperparameters)

Polyak.set_parameters(
    x0=[20,20],
    f=f1,
    iters=[50,100],
    f_star=0,
    eps=0.002)

Adam.set_parameters(
    x0=[20,20],
    f=f1,
    beta1=0.2,
    beta2=0.2,
    alpha=0.2,
    iters=[50,100],
    eps=0.002)

# outputs = run_optimisations(Polyak)
# a = outputs[0]
# del a['X']
# del a['Y']
# print(a)
# plot_y(outputs)
