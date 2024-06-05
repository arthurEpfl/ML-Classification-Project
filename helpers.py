"""Some helper functions for project 1."""
import csv
import numpy as np
import os
import random


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids

def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})



def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return np.exp(t) / (1 + np.exp(t))


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_for_one_fold(y, x, ratio, additional_zeros_y, additional_zeros_x, k_indices, k, model,
                                  model_hyper_parameters, evaluate_accuracy, evaluate_f1_score):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N, D)
        ratio: ration of the number of zeros to the number of ones in dataset
        additional_zeros_y: shape=(N,). Used for simuluating actual fraction of zeros to ones in dataset.
        additional_zeros_x:  shape=(N, D). Used for simuluating actual fraction of zeros to ones in dataset.
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        model:      model we use for training
        model_hyper_parameters: array of model hyper parameters
        evaluate_accuracy:       function we use for evaluating accuracy of the model
        evaluate_f1_score: function that evaluates f1 score for this model

    Returns:
        accuracy: test accuracy on the k-th fold
        f1_score: test f1_score on the k-th fold
    """

    # splitting the dataset on train and test
    # print(int(((9-ratio)*k_indices.shape[1])/(ratio+1)), k_indices.shape, (9-ratio))
    x_test = np.concatenate([x[k_indices[k]], additional_zeros_x[:int(((9-ratio)*k_indices.shape[1])/(ratio+1))]], axis=0)
    x_train = x[np.ravel(np.concatenate((k_indices[:k], k_indices[k+1:])))]
    y_test = np.concatenate([y[k_indices[k]], additional_zeros_y[:int(((9-ratio)*k_indices.shape[1])/(ratio+1))]], axis=0)
    y_train = y[np.ravel(np.concatenate((k_indices[:k], k_indices[k+1:])))]
    # fitting the model on train part
    w_star, loss = model(y_train, x_train, *model_hyper_parameters)
    # print(np.where(y_test == 0)[0].shape[0]/np.where(y_test == 1)[0].shape[0])

    return evaluate_accuracy(y_test, x_test, w_star), evaluate_f1_score(y_test, x_test, w_star), w_star


def cross_validation(y, x, ratio, additional_zeros_y, additional_zeros_x, k_fold, model, model_hyper_parameters, evaluate_accuracy, evaluate_f1_score):
    """

        y:          shape=(N,)
        x:          shape=(N, D)
        ration:     ration of the number of zeros to the number of ones in dataset
        additional_zeros_y: shape=(N,). Used for simuluating actual fraction of zeros to ones in dataset.
        additional_zeros_x:  shape=(N, D). Used for simuluating actual fraction of zeros to ones in dataset.
        k_fold:     integer, the number of folds
        model:      model we use for training
        model_hyper_parameters: array of model hyper parameters
        loss:       function we use for evaluating the model
        evaluate_f1_score: function that evaluates f1 score for this model

    Returns:
        accuracy: averaged accuracy over all fold
        f1_score: averaged f1_score over all fold
    """

    seed = 12
    k_fold = k_fold
    k_indices = build_k_indices(y, k_fold, seed)

    all_test_accuracy = []
    all_f1_scores = []
    all_w = []

    # averaging the losses over all folds
    for k in range(0, k_fold):
        test_accuracy, f1_score, w = cross_validation_for_one_fold(y,
                                                                   x,
                                                                   ratio,
                                                                   additional_zeros_y,
                                                                   additional_zeros_x,
                                                                   k_indices,
                                                                   k,
                                                                   model,
                                                                   model_hyper_parameters,
                                                                   evaluate_accuracy,
                                                                   evaluate_f1_score)

        print("test accuracy and f1 score respectively on ", k, "-th fold: ", test_accuracy, f1_score)
        all_w.append(w)
        all_test_accuracy.append(test_accuracy)
        all_f1_scores.append(f1_score)

    return np.mean(all_test_accuracy), np.mean(all_f1_scores), all_w[np.argmax(all_f1_scores)]


def fit_resample(X, y, fraction=1, seed=42):
    """
        Args:
            y:          shape=(N,)
            x:          shape=(N, D)
            fraction:   ratio of the number zeros to the number of ones in desired resampling
            seed:       seed for randomness

        Returns:
            (X_resampled, y_resampled): resampled data
            (X[additional_indices], y[additional_indices]): additional zero labeled data for stratification
    """
    np.random.seed(seed)
    unique_classes = list(set(y))
    num_samples = min(len(np.where(y == cls)[0]) for cls in unique_classes)

    indices_to_keep = []
    additional_indices = []
    for cls in unique_classes:
        indices = [i for i, label in enumerate(y) if label == cls]
        if cls == 1:
            selected_indices = random.sample(indices, num_samples)
        else:
            selected_indices = random.sample(indices, int(9*num_samples))
            additional_indices = selected_indices[int(fraction*num_samples):]
            selected_indices = selected_indices[:int(fraction*num_samples)]
        indices_to_keep.extend(selected_indices)

    X_resampled = X[indices_to_keep]
    y_resampled = y[indices_to_keep]

    return (X_resampled, y_resampled), (X[additional_indices], y[additional_indices])
