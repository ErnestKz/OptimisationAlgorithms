def finite_diff(f, x, delta=0.01):
    return (f(x) - f(x - delta)) / delta


def plot_y_vary_param(inputs, comparing):
    i = inputs[0]
    f = i['f']
    function_name = f.function_name
    function_latex = f.latex()
    
    opt_alg = i['algorithm']
    algorithm_name = opt_alg.algorithm_name

    hyperparams = (opt_alg.hyperparameters + ['x0'])
    hyperparams.remove(comparing)
    
    hs = hyperparams_string(i, hyperparams)
    top = (rf'{algorithm_name} on ${function_latex}$ varying {comparing}')
    title_string = top + " \n" +hs

    fig, ax = plt.subplots()
    ax.set_title(title_string)
    ax.set_ylabel(f'${function_name}$')
    ax.set_xlabel(r'$i$')

    rangei = 50
    legend_labels = []
    for (X, Y, var) in dicts_collect(("X", "Y", comparing), inputs):
        ax.plot(range(len(Y)), Y, linewidth=2.0)
        legend_labels += [(comparing + ": " + str(var))]
    ax.legend(legend_labels)
    return ax

def hyperparams_string(inputs, hyperparams):
    string = ""
    for p in hyperparams:
        string += f"{p}={inputs[p]}, "
    return string[0:-2]

def dicts_collect(keys, dicts):
    values = []
    for dict in dicts:
        values += [[dict[key] for key in keys]]
    return values
