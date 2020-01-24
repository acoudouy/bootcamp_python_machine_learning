import numpy as np
import math

def entropy(array):
    """Compute the Shannon Entropy of a non-empty np.ndarray"""
    if isinstance(array, np.ndarray) == 1:
        res = 0.0
        el_unique, count = np.unique(array, return_counts=True)
        nb_el_unique = len(el_unique)
        nb_el_tot = len(array)
        dict_count = dict(zip(el_unique, count))
        for i in dict_count:
            res += (dict_count[i] / nb_el_tot)  * math.log2((dict_count[i] / nb_el_tot))
        return(-res)
    else:
        print("Entropy: param not a np.ndarray")

