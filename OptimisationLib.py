def finite_diff(f, x, delta=0.01):
    return (f(x) - f(x - delta)) / delta
