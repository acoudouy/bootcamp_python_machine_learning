import numpy as np

def accuracy_score_(y_true, y_pred):
    """Compute the accuracy score. """
    res = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            res += 1
    res = res / len(y_true)
    return(res)

y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])

print(accuracy_score_(y_true, y_pred))


y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog',
'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet',
'dog', 'norminet'])
print(accuracy_score_(y_true, y_pred))
