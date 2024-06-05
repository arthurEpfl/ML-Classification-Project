from costs import *
from helpers import *
from decision_tree import *
from random_forest import *
# from GradientBoosting import *



def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, weight_for_one=1):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        weight_for_one: positive real number. This is a penalty for zero prediction
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    # Define parameter to store the initial loss# Define parameter to store the initial loss
    w = initial_w
    # # Adding identity column for w_0
    # tx = np.c_[np.ones((y.shape[0], 1)), tx]

    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient_mse(y, tx, w, weight_for_one)
        # update gradient
        w -= gamma * gradient
        # compute loss
        loss = compute_mse_loss(y, tx, w, weight_for_one)

        print(
            "GD iter. {bi}/{ti}: loss={l}, w={w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w,
            )
        )

    return w, compute_mse_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1, weight_for_one=1):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        weight_for_one: positive real number. This is a penalty for zero prediction
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """

    # Define parameter to store the initial loss
    w = initial_w
    # # Adding identity column for w_0
    # tx = np.c_[np.ones((y.shape[0], 1)), tx]

    for n_iter in range(max_iters):
        # generate minibatch
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # compute gradient
            gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w, weight_for_one)
            # update gradient
            w -= gamma * gradient
            # compute loss
            loss = compute_mse_loss(y, tx, w, weight_for_one)

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w={w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w,
            )
        )

    return w, compute_mse_loss(y, tx, w)


def least_squares(y, tx):
    """
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    # # Adding identity column for w_0
    # tx = np.c_[np.ones((y.shape[0], 1)), tx]

    w_star = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return w_star, compute_mse_loss(y, tx, w_star)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    # # Adding identity column for w_0
    # tx = np.c_[np.ones((y.shape[0], 1)), tx]

    N = y.shape[0]
    w_star = np.linalg.solve(tx.T.dot(tx) + np.eye(tx.shape[1]).dot(lambda_ * (2 * N)), tx.T.dot(y))
    return w_star, compute_mse_loss(y, tx, w_star)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )
        gamma: float

    Returns:
        w: shape=(D, )
        loss: scalar number
    """

    gradient = compute_gradient_log_loss(y, tx, w)
    w -= gamma * gradient

    return compute_log_loss(y, tx, w), w


def logistic_regression(y, tx, initial_w, max_iters, gamma):

    """The Logistic Regression with gradient descent (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        log_loss: scalar.
    """

    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    # tx = np.c_[np.ones((y.shape[0], 1)), tx]

    w = initial_w
    best_w = initial_w
    best_loss = float(1e9)
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        if loss < best_loss:
            best_w = w
            best_loss = loss
        # log info
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, compute_log_loss(y, tx, best_w)


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )
        gamma: scalar
        lambda_: scalar

    Returns:
        w: shape=(D, )
        loss: scalar number
    """
    gradient = compute_gradient_penalized_log_loss(y, tx, w, lambda_)
    loss = compute_penalized_log_loss(y, tx, w, lambda_)
    w -= gamma * gradient

    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The Regularized Logistic Regression with gradient descent (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        lambda_: scalar
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        log_loss: scalar.
    """

    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    # tx = np.c_[np.ones((y.shape[0], 1)), tx]

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, compute_log_loss(y, tx, w)


def decision_tree(y, tx, max_depth, criterion_name, one_prediction_threshold=0.5):
    """Decision Tree algotithm

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        max_depth: maximal depth of a tree,
        criterion_name: criterion we use for training
        one_prediction_threshold: threshold of predicting one
    Returns:
        tree: optimal trained tree
        (accuracy, f1_score): accuracy and F1 score on train dataset
    """

    tree = DecisionTree(max_depth=max_depth,
                        criterion_name=criterion_name)
    tree.fit(y.reshape((-1, 1)), tx)

    predicted_y = tree.predict_label(tx, one_prediction_threshold)

    return tree, (accuracy(y, predicted_y), F1_score(y, predicted_y))


def random_forest(y, tx, max_depth, criterion_name, n_trees, one_prediction_threshold=0.5):
    """Random Forest algotithm

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        max_depth: maximal depth of a tree,
        criterion_name: criterion we use for training
        n_trees: number of trees in forest
        one_prediction_threshold: threshold of predicting one
    Returns:
        forest: optimal trained random forest
        (accuracy, f1_score): accuracy and F1 score on train dataset
    """

    forest = RandomForest(max_depth=max_depth, criterion_name=criterion_name, n_trees=n_trees)

    forest.fit(y.reshape((-1, 1)), tx)

    predicted_y = forest.predict_label(tx, one_prediction_threshold)

    return forest, (accuracy(y, predicted_y), F1_score(y, predicted_y))