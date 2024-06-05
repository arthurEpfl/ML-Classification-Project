import numpy as np
import random


def entropy(y):
    """
    Computes entropy of the provided distribution. Used log(value + eps) for numerical stability

    Args:
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns:
    float
        Entropy of the provided subset
    """

    if y.shape[0] == 0:
        return 0

    EPS = 0.0005

    class_probabilities = np.zeros(y.shape[1])

    for cl in range(y.shape[1]):
        class_probabilities[cl] = len(np.where(y.T[cl] == 1)[0]) / y.shape[0]

    loss = 0
    for cl in range(y.shape[1]):
        loss -= (class_probabilities[cl] * np.log(class_probabilities[cl] + EPS))

    return loss


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Args:
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns:
    float : Gini impurity of the provided subset
    """
    if y.shape[0] == 0:
        return 0

    class_probabilities = np.zeros(y.shape[1])

    for cl in range(y.shape[1]):
        class_probabilities[cl] = len(np.where(y.T[cl] == 1)[0]) / y.shape[0]

    loss = 1
    loss -= np.sum(class_probabilities ** 2)

    return loss


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    def __init__(self, feature_index, threshold, proba=np.array([])):
        self.feature_index = feature_index
        self.threshold = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree:
    all_criterions = {
        'gini': gini,
        'entropy': entropy,
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False, RSM=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.RSM = RSM
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None
        self.debug = debug

    def make_split(self, feature_index, threshold, X, y):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Args:
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns:
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j >= threshold
        """
        # print(X.shape, feature_index)
        indexes_left = np.where(X.T[feature_index] < threshold)
        indexes_right = np.where(X.T[feature_index] >= threshold)

        return (X[indexes_left], y[indexes_left]), (X[indexes_right], y[indexes_right])

    def make_split_only_y(self, feature_index, threshold, X, y):
        """
        Split only target values into two subsets with specified feature and threshold

        Args:
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        indexes_left = np.where(X.T[feature_index] < threshold)
        indexes_right = np.where(X.T[feature_index] >= threshold)

        return y[indexes_left], y[indexes_right]

    def calculate_entropy(self, y, X, feature_index, threshold):
        """
            Args:
            y : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

            X : np.array of type float with shape (n_objects, n_features)

            Feature matrix representing the selected subset
            feature_index: feature index
            threshold: Threshold

            Returns:
                functionas: G(j, t) = criterion(y) - \frac{|L|}{|y|}criterion(L) -
                \frac{|R|}{|y|}criterion(R), where L, R is a partition of y relatively feature feature_index and threshold threshold


        """
        y_left, y_right = self.make_split_only_y(feature_index, threshold, X, y)
        L_size = y_left.shape[0]
        R_size = y_right.shape[0]
        assert (L_size + R_size == y.shape[0])

        return self.criterion(y) - L_size / y.shape[0] * self.criterion(y_left) - \
               R_size / y.shape[0] * self.criterion(y_right)

    def choose_best_split(self, X, y, feature_subspace):
        """
        Args:
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        feature_subspace: subspace of features we can use, needed for RSM

        Returns:
        feature_index : int
            Index of best feature to make split with

        threshold : float
            Best threshold value to perform split

        """
        # array of functional values on the given feature and threshold
        G = np.array([])
        # feature and threshold used above
        features = np.array([])
        thresholds = np.array([])

        for feature in feature_subspace:
            unique_values = np.unique(X.T[feature])
            np.append(unique_values, np.max(unique_values) + 1)
            for threshold in unique_values:
                features = np.append(features, feature)
                thresholds = np.append(thresholds, threshold)
                G = np.append(G, self.calculate_entropy(y, X, feature, threshold))

        print("choose best feature and threshold: ", features[np.argmax(G)], thresholds[np.argmin(G)])
        return int(features[np.argmax(G)]), thresholds[np.argmax(G)]

    def make_tree(self, X, y, depth):
        """
        Recursively builds the tree

        Args:
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        depth: depth of the current node

        Returns:
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        if self.RSM:
            # if we perform RSM, restrict our data to random subset of features
            best_feature, best_threshold = self.choose_best_split(X, y, np.random.choice(range(X.shape[1]),
                                                                size=random.randint(1, X.shape[1]), replace=False))
        else:
            best_feature, best_threshold = self.choose_best_split(X, y, np.arange(0, X.shape[1]))

        (X_left, y_left), (X_right, y_right) = self.make_split(best_feature, best_threshold, X, y)

        left_node = None
        right_node = None
        proba = np.array([])

        if depth < self.max_depth and y_left.shape[0] > 0 and y_right.shape[0] > 0:
            # recursion
            left_node = self.make_tree(X_left, y_left, depth + 1)
            right_node = self.make_tree(X_right, y_right, depth + 1)
        else:
            # if we can't perform a recursion or it is not profitable make root a leave
            class_probabilities = np.zeros(y.shape[1])

            for cl in range(y.shape[1]):
                class_probabilities[cl] = len(np.where(y.T[cl] == 1)[0]) / y.shape[0]

            proba = class_probabilities

        node = Node(best_feature, best_threshold, proba)

        node.left_child = left_node
        node.right_child = right_node

        return node

    def fit(self, y, X):
        """
            Args:

            y : np.array of type int with shape (n_objects, 1) in classification
                       of type float with shape (n_objects, 1) in regression
                Column vector of class labels in classification or target values in regression

            X : np.array of type float with shape (n_objects, n_features)
                Feature matrix representing the data to train on

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion = self.all_criterions[self.criterion_name]

        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y, 0)

        print("fitting finished")

    def predict_for_row_rec(self, node, row):
        """
            Predict the class probabilities using the provided data

            Args:
            node: vertex of a tree we are currently staying at
            row : np.array of type float with shape (1, n_features)
                Feature matrix representing the data the predictions should be provided for

            Returns:
            y_predicted_probs : np.array of type float with shape (n_classes, )
                Probabilities of each class for the provided objects

        """
        if node.proba.shape[0] != 0:
            return node.proba
        else:
            if row[node.feature_index] < node.threshold:
                return self.predict_for_row_rec(node.left_child, row)
            else:
                return self.predict_for_row_rec(node.right_child, row)

    def predict_proba_for_row(self, row):
        """
            Predict the class probabilities using the provided data

            Args:
            row : np.array of type float with shape (n_features, )
                vector representing the data the predictions should be provided for

            Returns:
            y_predicted_probs : np.array of type float with shape (n_classes, )
                Probabilities of each class for the provided objects

        """
        return self.predict_for_row_rec(self.root, row)

    def predict_proba(self, X):
        """
        Predict the class probabilities using the provided data

        Args:
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns:
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """

        y_predicted_probs = []

        for i, row in enumerate(X):
            y_predicted_probs.append(self.predict_proba_for_row(row))

        return np.array(y_predicted_probs)

    def predict_label_for_row(self, row, one_prediction_threshold=0.5):
        """
            Args:
            row : np.array of type float with shape (n_features, )
                vector representing the data the predictions should be provided for
            one_prediction_threshold: threshold of predicting one

            Returns:
            int : predicted class for this data point

        """
        predicted_proba = self.predict_proba_for_row(row)
        predicted_label = predicted_proba[1] > one_prediction_threshold
        return predicted_label

    def predict_label(self, X, one_prediction_threshold=0.5):
        """
            Args:
                X : np.array of type float with shape (n_objects, n_features)
                Feature matrix representing the data the predictions should be provided for
                one_prediction_threshold: threshold of predicting one
            Returns:
               np.array of shape (n_objects, ):  predicted class for this data point

       """
        predicted_proba = self.predict_proba(X)
        predicted_label = np.zeros(X.shape[0])
        predicted_label[np.where(predicted_proba.T[1] > one_prediction_threshold)] = 1

        return predicted_label
