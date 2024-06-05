import numpy as np

from decision_tree import *
import random


class RandomForest:

    def __init__(self, n_classes=None, n_trees=1, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        self.trees = []
        self.n_trees = n_trees

        # creating a list of n_trees trees with given parameters and True RSM flag for our forest
        for _ in range(n_trees):
            self.trees.append(DecisionTree(n_classes=n_classes, max_depth=max_depth, min_samples_split=min_samples_split,
                 criterion_name=criterion_name, debug=debug, RSM=True))

        self.trees = np.array(self.trees)

    def bootstrap(self, y, X):
        """
            Args:

                y : np.array of type int with shape (n_objects, 1) in classification
                           of type float with shape (n_objects, 1) in regression
                    Column vector of class labels in classification or target values in regression

                X : np.array of type float with shape (n_objects, n_features)
                    Feature matrix representing the data to train on

            Return:
                a new generated data as np.arrays of the same shape generated from y and X
        """

        X_data = []
        y_data = []

        # generating a copy of X and y from "the same" distribution as the original dataset
        for _ in range(self.n_trees):
            indices = random.choices(range(X.shape[0]), k=X.shape[0])
            X_data.append(X[indices])
            y_data.append(y[indices])

        return np.array(y), np.array(X)

    def fit(self, y, X):
        """
            Args:

            y : np.array of type int with shape (n_objects, 1) in classification
                       of type float with shape (n_objects, 1) in regression
                Column vector of class labels in classification or target values in regression

            X : np.array of type float with shape (n_objects, n_features)
                Feature matrix representing the data to train on

        """

        # fitting all trees in the forest using bootstraped data
        for tree in self.trees:
            y_bootstrap, X_bootstrap = self.bootstrap(y, X)
            tree.fit(y_bootstrap, X_bootstrap)

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

        predictions_for_each_tree = []
        for tree in self.trees:
            predictions_for_each_tree.append(tree.predict_proba_for_row(row))

        return np.mean(np.array(predictions_for_each_tree), axis=0)

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
        predictions_for_each_tree = []
        for tree in self.trees:
            predictions_for_each_tree.append(tree.predict_proba(X))

        return np.mean(np.array(predictions_for_each_tree), axis=0)

    def predict_label_for_row(self, row):
        """
            Args:
            row : np.array of type float with shape (n_features, )
                vector representing the data the predictions should be provided for

            Returns:
            int : predicted class for this data point

        """
        return np.argmax(self.predict_proba_for_row(row))

    def predict_label(self, X):
        """
            Args:
                X : np.array of type float with shape (n_objects, n_features)
                Feature matrix representing the data the predictions should be provided for

            Returns:
               np.array of shape (n_objects, ):  predicted class for this data point

       """
        return np.argmax(self.predict_proba(X), axis=1)


