import numpy as np
from entropy import entropy
from gini import gini


def information_gain(array_source, array_children_list, criterion='gini'):
    """Computes the information gain between the first and second array using the criterion 'gini' or 'entropy' 333"""
    if isinstance(array_source, np.ndarray) == 1 and isinstance(array_children_list, np.ndarray) == 1:
        if criterion == "gini" or criterion == "entropy":
            if criterion == "gini":
                So = gini(array_source)
                q = len(array_children_list)
                N = len(array_source)
                somme = 0.0
                for i in range(q):
                    somme += (len(array_children_list / N) * gini(array_children_list))
                IG = So - somme
                return(IG)
            else:
                So = entropy(array_source)
                q = len(array_children_list)
                N = len(array_source)
                somme = 0.0
                for i in range(q):
                    somme += (len(array_children_list / N) * entropy(array_children_list))
                IG = So - somme
                return(IG)
        else:
            print("info_gain: error in children list or criterion type")
    else:
        print("info_gain: error in type of array")

