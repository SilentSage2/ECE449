import numpy as np
import matplotlib.pyplot as plt

from hw3_utils import visualization, get_dataset_fixed


class Stump():
    def __init__(self, data, labels, weights):
        '''
        Initializes a stump (one-level decision tree) which minimizes
        a weighted error function of the input dataset.

        In this function, you will need to learn a stump using the weighted
        datapoints. Each datapoint has 2 features, whose values are bounded in
        [-1.0, 1.0]. Each datapoint has a label in {+1, -1}, and its importance
        is weighted by a positive value.

        The stump will choose one of the features, and pick the best threshold
        in that dimension, so that the weighted error is minimized.

        Arguments:
            data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
            labels: An ndarray with shape (n, ). Values are +1 or -1.
            weights: An ndarray with shape (n, ). The weights of each
                datapoint, all positive.
        '''
        # You may choose to use the following variables as a start

        # The feature dimension which the stump will decide on
        # Either 0 or 1, since the datapoints are 2D
        self.dimension = 0

        # The threshold in that dimension
        # May be midpoints between datapoints or the boundaries -1.0, 1.0
        self.threshold = -1.0

        # The predicted sign when the datapoint's feature in that dimension
        # is greater than the threshold
        # Either +1 or -1
        self.sign = 1

        # My code begins here
        loss = np.inf
        for k in range(np.shape(data)[1]):
            threshold_list = self.get_threshold(data[:,k])
            for j in range(len(threshold_list)):
                for s in [-1,1]:
                    if  loss > self.GetLoss(data[:,k], labels, threshold_list[j], s, weights):
                        loss = self.GetLoss(data[:,k], labels, threshold_list[j], s, weights)
                        self.threshold = threshold_list[j]
                        self.dimension = k
                        self.sign      = s

        pass

    # define a help function
    def GetLoss(self, x_k, y, t, s, weights):
        loss = 0
        for i in range(len(x_k)):
            y_hat = s
            if x_k[i] < t:
                y_hat = -s
            loss += weights[i]*(y_hat != y[i])
        return loss
    def get_threshold(self, data_k):
        list_sorted = data_k.copy()
        list_sorted.sort()
        result_list = []
        for i in range(len(list_sorted)-1):
            result_list.append((list_sorted[i]+list_sorted[i+1])/2)
        result_list = np.unique(result_list).tolist()
        result_list.append(1)
        result_list.append(-1)
        result_list = np.unique(np.array(result_list)).tolist()
        return result_list

    def predict(self, data):
        '''
        Predicts labels of given datapoints.

        Arguments:
            data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].

        Returns:
            prediction: An ndarray with shape (n, ). Values are +1 or -1.
        '''

        # My code begins here
        n = np.shape(data)[0]
        prediction = np.zeros(n)
        x_k = data[:,self.dimension]
        for i in range(n):
            prediction[i] = self.sign
            if x_k[i] < self.threshold:
                prediction[i] = -self.sign
        return prediction

        pass


def bagging(data, labels, n_classifiers, n_samples, seed=0):
    '''
    Runs Bagging algorithm.

    Arguments:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
        n_classifiers: Number of classifiers to construct.
        n_samples: Number of samples to train each classifier.
        seed: Random seed for NumPy.

    Returns:
        classifiers: A list of classifiers.
    '''
    classifiers = []
    n = data.shape[0]

    for i in range(n_classifiers):
        np.random.seed(seed + i)
        sample_indices = np.random.choice(n, size=n_samples, replace=False)

        # My code begins here
        data_train  = data  [sample_indices]
        label_train = labels[sample_indices]
        weight = [1.0] * label_train.size
        classifier = Stump(data_train, label_train, weight)
        classifiers.append(classifier)
        # pass

    return classifiers


def adaboost(data, labels, n_classifiers):
    '''
    Runs AdaBoost algorithm.

    Arguments:
        data: An ndarray with shape (n, 2). Values in [-1.0, 1.0].
        labels: An ndarray with shape (n, ). Values are +1 or -1.
        n_classifiers: Number of classifiers to construct.

    Returns:
        classifiers: A list of classifiers.
        weights: A list of weights assigned to the classifiers.
    '''
    classifiers = []
    weights = []
    n = data.shape[0]
    data_weights = np.ones(n) / n

    for i in range(n_classifiers):

        # My code begins here
        d = data.shape[1]
        classifier = Stump(data, labels, data_weights)
        classifiers.append(classifier)
        # get the weight of each classifier -- alpha
        error = 0 
        for j in range(n):
            error += data_weights[j]*(classifier.predict(data[j].reshape(-1,d))!=labels[j])
        # weight of each classifier
        alpha = 0.5*np.log((1-error)/error) 
        weights.append(alpha)
        # get the normalization sum -- z_t
        z_t = 0
        for j in range(n):
            z_t += data_weights[j]*np.exp(-alpha*labels[j]*classifier.predict(data[j].reshape(-1,d)))
        # update data_weight
        for j in range(n):
            data_weights[j] = data_weights[j]*np.exp(-alpha*labels[j]*classifier.predict(data[j].reshape(-1,d)))/z_t
        # pass

    return classifiers, weights


if __name__ == '__main__':
    data, labels = get_dataset_fixed()

    # You can play with the dataset and your algorithms here
    # classifier = Stump()
