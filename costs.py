import numpy as np
from helpers import *


def compute_mse_loss(y, tx, w, weight_for_one=1):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
        weight_for_one: positive real number. This is a penalty for zero prediction
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # compute loss by MSE
    e = y - tx.dot(w)
    weights_for_gradient = np.ones(y.shape[0])
    weights_for_gradient[np.where(y == 1)] = weight_for_one
    return 1 / (2 * y.shape[0]) * (e**2).T.dot(weights_for_gradient.T)


def accuracy(y, y_pred):
    true_positives = sum(1 for true, pred in zip(y, y_pred) if true == 1 and pred == 1)
    true_negatives = sum(1 for true, pred in zip(y, y_pred) if true == 0 and pred == 0)
    return (true_positives + true_negatives) / y.shape[0]


def F1_score(y, y_pred):
    true_positives = sum(1 for true, pred in zip(y, y_pred) if true == 1 and pred == 1)
    false_positives = sum(1 for true, pred in zip(y, y_pred) if true == 0 and pred == 1)
    false_negatives = sum(1 for true, pred in zip(y, y_pred) if true == 1 and pred == 0)
    if true_positives + false_positives != 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0

    if (true_positives + false_negatives) != 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0

    if precision + recall == 0:
        return 0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def compute_gradient_mse(y, tx, w, weight_for_one=1):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        weight_for_one: positive real number. This is a penalty for zero prediction

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx.dot(w)
    # gradient = np.concatenate(([-np.mean(e)], [-np.mean(e * tx[:, i]) for i in range(1, w.shape[0])]))
    weights_for_gradient = np.ones(y.shape[0])
    weights_for_gradient[np.where(y == 1)] = weight_for_one
    gradient = np.array([-np.mean(e * tx[:, i] * weights_for_gradient.T) for
                                                                    i in range(w.shape[0])])
    return gradient


def compute_log_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    loss = 0
    for i in range(y.shape[0]):
        loss += (y[i] * np.log(sigmoid(tx[i].dot(w))) + (1 - y[i]) * np.log(1 - sigmoid(tx[i].dot(w))))
    return - 1/y.shape[0] * loss


def compute_gradient_log_loss(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a vector of shape (D, 1)
    """

    gradient = - 1 / y.shape[0] * tx.T.dot(y - sigmoid(tx.dot(w)))
    return gradient


def compute_penalized_log_loss(y, tx, w, lambda_):
    """
        Args:
            y:  shape=(N, )
            tx: shape=(N, D)
            w:  shape=(D, )
            lambda_: scalar

        Returns:
            loss: number
        """
    return compute_log_loss(y, tx, w) + lambda_ * np.linalg.norm(w)**2


def compute_gradient_penalized_log_loss(y, tx, w, lambda_):
    """
        Args:
            y:  shape=(N, )
            tx: shape=(N, D)
            w:  shape=(D, )
            lambda_: scalar

        Returns:
            gradient: shape=(D, )
        """

    return compute_gradient_log_loss(y, tx, w) + 2 * lambda_ * w



