# Functions that will be optimised:
# - Allows access to 
#   - Parital Derivatives
#   - String representation of the function (latex)
# - Constructor uses sympy to obtain the above

from sympy import simplify, latex, lambdify
import numpy as np

class OptimisableFunction:
    def __init__(self, sympy_function, sympy_symbols, function_name):
        self.sympy_symbols = sympy_symbols
        self.function_name = function_name
        
        self.sympy_function = sympy_function
        self.function = lambdify(sympy_symbols, sympy_function, modules="numpy")

        self.sympy_partial_derivatives = [sympy_function.diff(symbol) for symbol in sympy_symbols]
        self.partial_derivatives = [lambdify(sympy_symbols, p, modules="numpy") for p in self.sympy_partial_derivatives]

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


