import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['figure.facecolor'] = '1'
import matplotlib.pyplot as plt

from sympy import diff, lambdify, symbols, init_printing, Max, Abs
init_printing(use_unicode=False)


x, y = symbols('x y', real=True)

f1 = 3 * (x-9)**4 + 5 * (y-9)**2
df1dx = diff(f1, x)
df1dy = diff(f1, y)

f2 = Max(x-9 ,0) + 5 * Abs(y-9)
df2dx = diff(f2, x)
df2dy = diff(f2, y)

df2dxl = lambdify([x, y], df2dx, modules="numpy")
df2dyl = lambdify([x, y], df2dy, modules="numpy")
f2l = lambdify([x, y], f2, modules="numpy")
print(df2dxl(8, 1))
print(df2dyl(8, 8))

df1dxl = lambdify([x, y], df1dx, modules="numpy")
df1dyl = lambdify([x, y], df1dy, modules="numpy")
f1l = lambdify([x, y], f1, modules="numpy")


def polyak(x0, f, f_star, eps=0.0001, iters=50):
    x = x0
    X = [x]
    dfs = f.dfs
    f = f.f
    Y = [f(*x)]

    # now need to return x, y, dx
    for _ in range(iters):
        fdif = f(*x) - f_star
        df_squared_sum = np.sum(np.array([df(*x)**2 for df in dfs]))
        alpha = fdif / (df_squared_sum + eps)
        x = x - alpha * np.array([df(*x) for df in dfs])

        X += [x]
        Y += [f(*x)]
        # can  calculate by taking difference of the previous X
        
    return X, Y


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


def rmsprop(x0, f, alpha0, beta, eps, iters_max):
    sum = np.array([0.0, 0.0]) ; x = x0 ; alpha = alpha0

    dfs = f.dfs
    f = f.f
    
    X = [x]
    Y = [f(*x)]
    for _ in range(iters_max):
      x = x - (alpha * np.array([df(*x) for df in dfs]))
      sum = beta * sum + (1 - beta) * np.array([float(df(*x)**2) for df in dfs]) # would be nice to write out the formula for this
      alpha = alpha0 / (np.sqrt(sum) + eps)
      X += [x]
      Y += [f(*x)]
    return X, Y

def heavy_ball(x0, f, alpha, beta, iters_max):
    dfs = f.dfs
    f = f.f
    x = x0; z = np.array([0, 0]);

    X = [x]
    Y = [f(*x)]
    for _ in range(iters_max):
        z = beta * z + alpha * np.array([float(df(*x)) for df in dfs])
        x = x - z

        X += [x]
        Y += [f(*x)]
        
    return X, Y

def adam(x0, f, eps, beta1, beta2, alpha, iters_max):
    x = x0; m = np.array([0.0, 0.0]); v = np.array([0.0, 0.0])

    dfs = f.dfs
    f = f.f
    X=[x]
    Y=[f(*x)]
    
    for k in range(iters_max):
        i = k + 1
        # m = beta1 * m + np.sum((1 - beta1) * np.array([float(df(*x)) for df in dfs]))

        m = beta1 * m + (1 - beta1) * np.array([float(df(*x)) for df in dfs])
        v = beta2 * v + (1 - beta2) * np.array([float((df(*x)**2)) for df in dfs])
        
        mhat = (m / (1 - beta1**i))   # what are these doing?
        vhat = (v / (1 - beta2**i))
        # x = x - alpha * np.array([(mhat / (np.sqrt(vhati) + eps)) for vhati in vhat])
        x = x - alpha * (mhat / (np.sqrt(vhat) + eps))

        
        X += [x]
        Y += [f(*x)]

    return X,Y



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


f2 = Fn(f=f2l, dfs=[ df2dxl, df2dyl ], name="blah")


# polyak_alg = OptAlg(f=polyak, name="polyak")
# inputs= mkInput(
#     x0=[21, 21],
#     f=f2,
#     f_star=0,
#     eps=0.0001,
#     iters=60,
# )
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



# f2 = Fn(f=f2l, dfs=[ df2dxl, df2dyl ], name="blah")
# adam_alg = OptAlg(f=adam, name="adam")

# inputs= mkInput(
#     x0=[35,-20],
#     f=f2,
#     eps=0.0001,
#     beta1=0.9,
#     beta2=0.9,
#     alpha=0.4,
#     iters_max=70
# )

# out = runOptimisations(optAlg=adam_alg,
#                        inputs=inputs
#                        )
# print(out[0]['X'])
# print(out[0]['X'][-1])
# print(out[0]['Y'][-1])
# print(out[0]['Y'])

def hyperparams_string(inputs):
    string = ""
    for key, value in inputs.items():
        string += f"{key}={value}, "
    return string[0:-2]


# sepcify dict keys that you want the value to be collected
# it returns a list of tuples of the arugments
def dicts_collect(keys, dicts):
    values = []
    for dict in dicts:
        values += [[dict[key] for key in keys]]
    return values

def ploty(inputs, comparing):
    inp = inputs[0].copy()
    f_name = inp['f'].name
    optname = inp['algorithm']
    
    del inp[comparing]
    del inp['f']
    del inp['X']
    del inp['Y']
    inp['algorithm']
    hs = hyperparams_string(inp)

    title_string = optname + "; " + f_name + "; Varying " + comparing + " \n" + hs
    
    fig, ax = plt.subplots()
    ax.set_ylabel(r'$y(x_i)$')
    ax.set_xlabel(r'$i$')
    ax.set_title(title_string)
    
    rangei = 50
    legend_labels = []
    for (X, Y, var) in dicts_collect(("X", "Y", comparing), inputs):
        ax.plot(range(len(Y)), Y, linewidth=2.0)
        legend_labels += [(comparing + ": " + str(var))]
    ax.legend(legend_labels)

# ploty(out, 'eps')
