from Function import OptimisableFunction
from Algorithms import Polyak, Adam, HeavyBall, RMSProp
from Datatype import make_input, run_optimisations
from Plotting import plot_y

from sympy import symbols, Max, Abs

x1, x2 = symbols('x1 x2', real=True)
sym_f1 = 3 * (x1-9)**4 + 5 * (x2-9)**2
f1 = OptimisableFunction(sym_f1, [x1, x2], "f_1")

sym_f2 = Max(x1-9 ,0) + 5 * Abs(x2-9)
f2 = OptimisableFunction(sym_f2, [x1, x2], "f_2")

algs = [Polyak, Adam, HeavyBall, RMSProp]
for a in algs:
    print(a.algorithm_name, a.hyperparameters)

f2_standard = make_input(x0=[20,20],
                         f=f2,
                         iters=50)[0]

f1_standard = make_input(x0=[20,20],
                         f=f1,
                         iters=50)[0]

polyak_standard = make_input(f_star=0,
                             eps=0.001)[0]

polyak_data = make_input(**f2_standard, **polyak_standard)
run_optimisations(Polyak, polyak_data)
plot_y(polyak_data)
