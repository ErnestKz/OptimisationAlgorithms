import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['figure.facecolor'] = '1'
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from OptimisationAlgorithmToolkit.DataType import create_labels, get_titles

from matplotlib.ticker import LogLocator

import numpy as np

def plot_contour(records, x1r, x2r, log=False, sym=False):
    create_labels(records)
    t = get_titles(records)

    f = records[0]['f']
    
    X1, X2 = np.meshgrid(x1r, x2r)
    Z = np.vectorize(f.function)(X1, X2)
    if log:
        plt.contourf(X1, X2, Z, locator=LogLocator(), cmap=plt.get_cmap('gist_earth'))
    else:
        plt.contourf(X1, X2, Z, cmap=plt.get_cmap('gist_earth'))
    xlim = plt.xlim()
    ylim = plt.ylim()
    for (X, label) in dicts_collect(("X", "label"), records):
        plt.plot(X.T[0], X.T[1], linewidth=2.0, label=label)

    f = records[0]['f']
    function_name = f.function_name
    if sym:
        f_latex = f.latex()
        title = rf'${f_latex}$' + " \n " + title_string(records)
    else:
        title = title_string(records)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title)
    
        
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.colorbar()

def plot_path(records, xr):
    create_labels(records)
    f = records[0]['f'].function;
    function_name = records[0]['f'].function_name
    f_latex = records[0]['f'].latex()
    
    yr = [f(x) for x in xr]
    plt.plot(xr, yr)
    xlim = plt.xlim()
    ylim = plt.ylim()
    
    for (X, label) in dicts_collect(("X", "label"), records):
        xs = X.flatten()
        ys = [f(x) for x in xs]
        plt.plot(xs, ys, linewidth=2.0, label=label)

    plt.xlim(xlim)
    plt.ylim(ylim)    
    plt.legend()
    title = rf'${f_latex}$' + "\n" + title_string(records)
    plt.title(title)
    plt.ylabel(f'${function_name}$')
    plt.xlabel(r'$x$')

def plot_step_size(records, mean=True):
    create_labels(records)
    fig, ax = plt.subplots()
    f_latex = records[0]['f'].latex()
    for (X, label) in dicts_collect(("X", "label"), records):
        if mean:
            s = np.array([np.mean(x) for x in  step_sizes(X).T])
            ax.plot(np.arange(1, len(s)+1), s, linewidth=2.0, label=label)
        else:
            sX = step_sizes(X)
            for i in range(len(sX)):
                x = i + 1
                s = sX[i]
                ax.plot(np.arange(1, len(s)+1), s, linewidth=2.0, label=label + f' $x_{x} step$')
    ax.legend()

    title = rf'${f_latex}$' + " \n " + title_string(records)
    if mean:
        ax.set_title("Mean Step Across x's \n" + title)
    else:
        ax.set_title("Mean Step Across x's \n" + title)
    ax.set_ylabel(f'Step Size')
    ax.set_xlabel(r'$i$')
    

def title_string(records):
    title = ""
    t = get_titles(records)
    for _, v in t.items():
        title += v + '\n'
    return title
    
# [[x11 x21 x31 ...] [x12 x22 x32 ...] ...]  -> [[x12-x11 x13-x12 ...] [x22-x21 x23-x22 ...] ...]
def step_sizes(X):
    return np.array([(x[1:] - x[:-1]) for x in X.T])
        


def ploty(records, sym=False):
    create_labels(records)
    t = get_titles(records)
    
    fig, ax = plt.subplots()
    for (X, Y, label) in dicts_collect(("X", "Y", "label"), records):
        ax.plot(range(len(Y)), Y, linewidth=2.0, label=label)


    f = records[0]['f']
    function_name = f.function_name
        
    if sym:
        f_latex = f.latex()
        title = rf'${f_latex}$' + " \n " + title_string(records)
    else:
        title = title_string(records)
    
    ax.set_title(title)
    ax.set_ylabel(f'${function_name}$')
    ax.set_xlabel(r'$i$')
    
    ax.legend()
    return ax

def dicts_collect(keys, dicts):
    values = []
    for dict in dicts:
        values += [[dict[key] for key in keys]]
    return values
