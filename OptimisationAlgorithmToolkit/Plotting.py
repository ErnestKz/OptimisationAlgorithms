# - different optimisation algorithms themselves
#   - different hyper parameters
    
# - different functions to optimise
#   - differnet modifiers on those functions
#   - functions of different dimensions

# - different plots
# - 1 gradient
#   - x vs i
#   - f vs i
#   - tracing where it goes on the function
#     - plot the function itself
# - 2 gradients
#   - contour plot
#     - lines of where it steps are taken
#     - perhaps an animation
# - multiple gradients
#   - toggle through pairs of them
#     - treat as 2 gradients



def plot_y(inputs, comparing):
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

def hyperparams_string(inputs):
    string = ""
    for key, value in inputs.items():
        string += f"{key}={value}, "
    return string[0:-2]

def dicts_collect(keys, dicts):
    values = []
    for dict in dicts:
        values += [[dict[key] for key in keys]]
    return values
