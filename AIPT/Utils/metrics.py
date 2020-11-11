import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score

def binary_classification_metrics(outputs, labels, verbose=True):
    '''
    Find confusion matrix, accuracy, mcc.

    Args:
        outputs (iterable of float): network-inferred outputs
        labels (iterable of int in {0,1}): true class labels

    Returns: Triple of (confusion matrix, accuracy, mcc).

    # todo: cleaner to return a dictionary
    '''
    y_pred = []
    for a in outputs:
        if a[0] > a[1]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    y_true = labels.flatten()
    y_pred = np.array(y_pred)

    mat = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    if verbose:
        print(mat)
        print('Accuracy = %.3f, MCC = %.3f' % (acc, mcc))
    return mat, acc, mcc
