import numpy as np

def gini(array):
    """Computes the gini impurity of a np.ndarray"""
    if isinstance(array, np.ndarray) == 1:
        res = 0.0
        el_unique, count = np.unique(array, return_counts=True)
        nb_el_unique = len(el_unique)
        nb_el_tot = len(array)
        dict_count = dict(zip(el_unique, count))
        for i in dict_count:
            res += (dict_count[i] / nb_el_tot) ** 2
        return(1 - res)
    else:
        print("gini: param not a np.ndarray")

a = np.array([0,0,0,0,0])
print(gini(a))
a = np.array([6])
print(gini(a))
a = np.array(['a','a','b','b'])
print(gini(a))
a = np.array([0,0,1,0,'bob',1])
print(gini(a))
a = np.array([0,0,1,0,2,1])
print(gini(a))
a = np.array([0,1,2])
print(gini(a))
a = np.array([0,0,2])
print(gini(a))


