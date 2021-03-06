import numpy as np
from entropy import entropy
from gini import gini
from node import Node
from information_gain import information_gain


#class Node:
#    def __init__(self, data=None, labels=None,
#                 is_leaf=False, split_feature=None, split_kind=None, split_criteria=None,
#                 left=None, right=None,
#                 depth=0):

class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=10):
    """
    :param str criterion: 'gini' or 'entropy'
    :param max_depth: max_depth of the tree (Decision tree creation
    stops splitting a node if node.depth >= max_depth)
    """
        self.root = None # Root node of the tree
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
    """
    Build the decision tree from the training set (X, y). The training
    set has m data_points (examples).
    Each of them has n features.
    :param pandas.Dataframe X: Training input (m x n)
    :param pandas.Dataframe y: Labels (m x 1)
    :return object self: Trained tree
    """

