def finite_diff(f, x, delta=0.01):
    return (f(x) - f(x - delta)) / delta

class Function():
    def __init__(self, func, list_of_variables):
        self.func = func 
        self.list_of_variables = list_of_variables
    
    def _create_tuple_list(self, list_of_numbers):
        
        if len(list_of_numbers) != len(self.list_of_variables):
            raise 'Provide exact number of elements'
        
        list_of_tuples = []
        for i in range(len(list_of_numbers)):
            list_of_tuples.append((self.list_of_variables[i], list_of_numbers[i]))
        
        return list_of_tuples
            
    def fn(self, list_of_numbers):
        #takes in a list of tuples
        
        list_of_tuples = self._create_tuple_list(list_of_numbers)
            
        return self.func.subs(list_of_tuples)
    
    def df(self, wrt, list_of_numbers):
        
        list_of_tuples = self._create_tuple_list(list_of_numbers)
        
        #returns the value of derivative of function at x = number
        # can make this faster by precomputing gradients wrt list of variables but oh well lol
        return diff(self.func, wrt).subs(list_of_tuples)

class Fn:
    def __init__(self, f, dfs, name):
        self.f = f
        self.dfs = dfs
        self.name = name

class OptAlg:
    def __init__(self, f, name):
        self.f = f
        self.name = name

def mkInput(**kwargs):
    # for each input, we are expecting a certain shape
    # x0 : is an array, if an array of arrays then we have multiple instances
    # iters : is a scalar, if array then have multiple instances
    # f : is the function by default will be single value for now
    # the rest: scalars, otherwise multiple instances
    expected_vector = {"x0"}
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

def runOptimisations(optAlg, inputs):
    outputs = []
    for input in inputs:
        X, Y = optAlg.f(**input)
        input["X"] = X
        input["Y"] = Y
        # input["dX"] = dX
        input["algorithm"] = optAlg.name
        outputs += [input]
    return outputs


# out = runOptimisations(optAlg=polyak_alg,
#                        inputs=inputs
#                        )



# rms_alg = OptAlg(f=rmsprop, name="rmsprop")

# inputs= mkInput(
#     x0=[35,-20],
#     f=f2,
#     alpha0=0.4,
#     beta=0.9,
#     eps=0.0001,
#     iters_max=70
# )

# out = runOptimisations(optAlg=rms_alg,
#                        inputs=inputs
#                        )


# heavy_alg = OptAlg(f=heavy_ball, name="heavy ball")

# inputs= mkInput(
#     x0=[35,-20],
#     f=f2,
#     alpha=0.4,
#     beta=0.9,
#     iters_max=70
# )

# out = runOptimisations(optAlg=heavy_alg,
#                        inputs=inputs
#                        )
