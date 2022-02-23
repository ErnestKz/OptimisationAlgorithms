# Datatype.py

# Contains the record format that identifies metadata of optimisation runs.

import numpy as np

def make_input(**kwargs):
    expected_vector = { "x0" }
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

def run_optimisations(optimisation_algorithm, inputs):
    for input in inputs:
        input["X"], input["Y"] = optimisation_algorithm.algorithm(**input)
        input["algorithm"] = optimisation_algorithm
