# Functions that will be optimised:
# - Allows access to 
#   - Parital Derivatives
#   - String representation of the function (latex)
# - Constructor uses sympy to obtain the above

from sympy import simplify, latex, lambdify
import numpy as np

class BatchedFunction:
    def __init__(self, f, M, name="f"):
        self.f = f
        self.function = lambda  x1, x2 : f(np.array([x1,x2]), minibatch=M)
        self.M = M
        self.function_name = name
    
class FunctionIterator:
    # b = len(M) will behave like normal gradient descent
    def __init__(self, f, b, i):
        self.i = i
        self.f = f
        self.function = f.function
        if type(f) is SymbolicFunction:
            self.batch = False
        else:
            self.batch = True
            self.M = f.M
            self.m = len(self.M)
            if b is None:
                self.b = len(self.M) # act as non stochastic
            else:
                self.b = b

    def __iter__(self):
        self.epoch = -1
        self.batch_start_indices = iter(())
        return self

    def __next__(self):
        if (self.i <= 0):
            raise StopIteration
        self.i -= 1
        if not self.batch:
            return self.function, f.partial_derivatives
        
        self.batch_index = next(self.batch_start_indices, None)
        if self.batch_index == None:
            self.epoch += 1
            np.random.shuffle(self.M)
            self.batch_start_indices = iter(np.arange(0, (self.m-self.b)+1, self.b))
            self.batch_index = next(self.batch_start_indices, None)

        N = np.arange(self.batch_index, self.batch_index + self.b)
        fN = lambda x: self.f.f(x, minibatch=self.M[N])
        dfs = [(lambda x1, x2, xi=i : finite_diff(fN, np.array([x1, x2]), xi)) for i in range(2)]
        return fN, dfs
    
class SymbolicFunction:
    def __init__(self, sympy_function, sympy_symbols, function_name):
        self.sympy_symbols = sympy_symbols
        self.function_name = function_name
        
        self.sympy_function = sympy_function
        self.function = lambdify(sympy_symbols, sympy_function, modules="numpy")

        self.sympy_partial_derivatives = [sympy_function.diff(symbol) for symbol in sympy_symbols]
        self.partial_derivatives = [lambdify(sympy_symbols, p, modules="numpy") for p in self.sympy_partial_derivatives]

    def __iter__(self):
        return self

    def __next__(self):
        return self.function, self.partial_derivatives

    def __parameters_string(self):
        s = map(latex, self.sympy_symbols)
        return ",".join(s)
        
    def latex(self):
        return self.function_name + "(" + self.__parameters_string() + ") = " + latex(simplify(self.sympy_function))

    def partials_latex(self):
        s = map(latex, self.sympy_symbols)
        z = zip(self.sympy_partial_derivatives, s)
        return [ "\\frac{\\partial " +  self.function_name + "}{\\partial " + partial_wrt_name + "}" "=" + latex(simplify(partial))
                for (partial, partial_wrt_name) in z]

    def print_partials_latex(self):
        for p in self.partials_latex():
            print(p)


def finite_diff(f, x, i, delta=0.0001):
    d = np.zeros(len(x)) ; d[i] = delta
    return (f(x) - f(x - d)) / delta
