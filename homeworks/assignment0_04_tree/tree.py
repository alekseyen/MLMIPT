import queue

import numpy as np
from sklearn.base import BaseEstimator

def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    probas = np.mean(y, axis=0)

    return -np.sum(probas * np.log(probas + EPS))


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    probas = np.mean(y, axis=0)

    return 1 - np.sum(probas ** 2)


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """

    # YOUR CODE HERE
    return np.var(y)


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self, feature_index=None, threshold=None, left_proba=0, right_proba=0):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_proba = left_proba
        self.right_proba = right_proba
        self.left_child = None
        self.right_child = None

        #  Только для листовых вершин)
        self.is_leaf = False
        self.y_values = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True),  # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.n_classes = n_classes  # нужно только для задачи классификации
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE

        X_left = X_subset[X_subset[:, feature_index] < threshold]
        y_left = y_subset[X_subset[:, feature_index] < threshold]
        X_right = X_subset[X_subset[:, feature_index] >= threshold]
        y_right = y_subset[X_subset[:, feature_index] >= threshold]

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
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

        # YOUR CODE HERE

        return y_subset[X_subset[:, feature_index] < threshold], y_subset[X_subset[:, feature_index] >= threshold]

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # перебираем все возможные признаки и значения порогов
        # находим оптимальный признак и оптимальное значение порога

        best_criterion_value = np.inf
        feature_index, best_threshold = 0, 0  # best feature and threshold
        sample_size, feature_count = X_subset.shape

        for feature_id in range(feature_count):
            for cur_threshold in np.sort(np.unique(X_subset.T[feature_id]))[1:-1]:
                y_l, y_r = self.make_split_only_y(feature_id, cur_threshold, X_subset, y_subset)
                L = len(y_l) / len(X_subset) * self.criterion(y_l) + \
                    len(y_r) / len(X_subset) * self.criterion(y_r)

                if L < best_criterion_value:
                    best_criterion_value = L
                    feature_index = feature_id
                    best_threshold = cur_threshold

        return feature_index, best_threshold

    def make_tree(self, X_subset, y_subset, depth=0):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        # улсовия остановки рекурсии
        if depth == self.max_depth or len(y_subset) < self.min_samples_split:
            leaf_vertex = Node()
            leaf_vertex.is_leaf = True

            if self.classification:  # если задача классификации
                leaf_vertex.y_values = np.mean(y_subset, axis=0)
            elif self.criterion == 'variance':
                leaf_vertex.y_values = np.mean(y_subset)
            else:
                leaf_vertex.y_values = np.median(y_subset)

            return leaf_vertex

        # print(*self.choose_best_split(X_subset, y_subset))
        new_node = Node(*self.choose_best_split(X_subset, y_subset))
        (X_l, y_l), (X_r, y_r) = self.make_split(new_node.feature_index, new_node.threshold, X_subset, y_subset)

        new_node.left_child = self.make_tree(X_l, y_l, depth + 1)
        new_node.right_child = self.make_tree(X_r, y_r, depth + 1)

        return new_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def predict(self, X):
        """
        Predict the target value or class label the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """

        # YOUR CODE HERE
        n_objects = X.shape[0]
        if self.classification:
            y_pred = np.argmax(self.predict_proba(X), axis=1).reshape(n_objects, 1)
        else:
            n_objects, _ = X.shape
            y_pred = np.zeros((n_objects, 1))
            indices = np.arange(n_objects)
            q_ = queue.Queue()
            q_.put((indices, X, self.root))
            while not q_.empty():

                indices_subset, X_subset, node = q_.get()
                if node.is_leaf:
                    y_pred[indices_subset] = node.y_values
                else:
                    (X_l, y_l), (X_r, y_r) = self.make_split(node.feature_index, node.threshold,
                                                             X_subset, indices_subset)
                    q_.put((y_l, X_l, node.left_child))
                    q_.put((y_r, X_r, node.right_child))

        return y_pred

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE

        y_predicted_probas = np.zeros((X.shape[0], self.n_classes))
        indices_ = np.arange(X.shape[0])
        q_ = queue.Queue()
        q_.put((indices_, X, self.root))
        while not q_.empty():
            indices_subset, X_subset, node = q_.get()
            if node.is_leaf:
                y_predicted_probas[indices_subset] = node.y_values
                continue

            (X_l, y_l), (X_r, y_r) = self.make_split(node.feature_index, node.threshold,
                                                     X_subset, indices_subset)
            q_.put((y_l, X_l, node.left_child))
            q_.put((y_r, X_r, node.right_child))

        return y_predicted_probas
