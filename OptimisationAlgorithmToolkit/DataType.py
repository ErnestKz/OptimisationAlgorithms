# Each record should contain its label depending on what are the other records in the list.

# The user semi-mannually inputs what the title should be.
# - Have utility functions to extract pieces of the title from the list of records.

# Function that takes in a list of records.
#  - For each record determines the label based on what is in the list of records.

# Perhaps there should be a function that calculatesthe meta information that is used by both
# - utility functions that extract peieces of title
# - function that assigns the labels to each individual record


# MetaInfo: extracts:
# - Which optimisaiton functions there area
# - For each optimisation function
#   - What are the parameters that are not varying and what values do they have
#   - What are the parameters that are varying and what values do they have




# {
#   ...
#   ...
#   label: 
# }
# label made up from what uniquely identifies it
# - first is optimisation algorithm itself
# - second are the hyperparmeters that uniqely identifies the cluster of algorithms
#   - RMSProp alpha0=0.4
#   - RMSProp alpha0=0.5
#   - Adam    beta1=0.2  beta2=0.4
#   - Adam    beta1=0.3  beta2=0.5

# - Then would like to extract the common descriptive pieces
#   - Different common pieces per algorithm used
#     - Records -> AlgorihtmName -> CommonThingsString
#       - Adam: beta1=0.1 eps=0.0001 iters=50 x0=[1, 1]
#       - RMSProp:  eps=0.0001 iters=50 x0=[1, 1]


# MetaRecord extracts
# - Algorithms and their corresponding Varying fields
# {
#   "Adam"    : ["eps", "beta1"]
#   "RMSProp" : ["eps", "alpha0"]
# }


# meta_record = meta(inputs)
# inputs = create_labels(meta_record, inputs)
# inputs = get_title(meta_record, inputs)

# get_titles returns
# {
#   "Adam" : "Adam: beta1=0.1 eps=0.0001 iters=50 x0=[1, 1]",
#   "RMSProp" : "RMSProp:  eps=0.0001 iters=50 x0=[1, 1]"
# }

import numpy as np

def get_titles(records):
    m = meta(records)
    t = {}
    for alg_name in m.keys():
        t[alg_name] = get_title(alg_name, records, m)
    return t
    
def get_title(alg_name, records, meta):
    title = f'{alg_name}:'
    algs = alg(records, alg_name)

    r = algs[0]
    params = set(r["algorithm"].all_parameters)
    varied = meta[alg_name]
    params.remove('f')
    params = params - varied
    
    for p in params:
        if p in r:
            title += f' {p}={r[p]}'
    return title

def create_labels(records):
    m = meta(records)
    for r in records:
        r['label'] = create_label(r, m)

# e.g: Adam    beta1=0.2  beta2=0.4
def create_label(record, meta):
    alg_name = record['algorithm'].algorithm_name
    differing_fields = meta[alg_name]
    label = f'{alg_name}'
    for f in differing_fields:
        label += f' {f}={record[f]}'
    return label

# {
#   "Adam"    : ["eps", "beta1"]
#   "RMSProp" : ["eps", "alpha0"]
# }
def meta(records):
    mr = {}
    algs = get_algs(records)
    for a in algs:
        a_records = alg(records, a)
        mr[a] = differing_fields(a_records)
    return mr

def differing_fields(records):
    diff_fields = set({})
    t = records[0]
    for r in records:
        for key, value in r.items():
            # print("a")
            # print(t[key])
            # print(type(value))
            # print(isinstance(value, list))
            
            if isinstance(value, list):
                value = np.array(value)
            if isinstance(t[key], list):
                t[key] = np.array(t[key])
                
            b = t[key] == value
            # print(b)
            # print(type(b))
            if type(b) == np.ndarray:
                b = b.all()
            if not (b):
                diff_fields.add(key)
            

    diff_fields.discard('X')
    diff_fields.discard('Y')
    return diff_fields

# extract one algorithm type, filter out the rest
def alg(records, algorithm_name):
    return list(filter(lambda r: r['algorithm'].algorithm_name == algorithm_name, records))

# gets algorithms names in the records
def get_algs(records):
    algs = set({})
    for r in records:
        algs.add(r['algorithm'].algorithm_name)
    return algs
    
    
# wonder how this would look in haskell
# funcitonal operators and stuff, would it make it easier.
