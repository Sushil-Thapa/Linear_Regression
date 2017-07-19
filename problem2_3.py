import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import scale
from numpy.distutils.system_info import p

from sklearn import linear_model
import sys


class LinearRegressor:
    def __init__(self, numOfFeatures=2, iterationValue=100, alphaValue=0.0001, of=None):
        self.out = of
        self.numOfFeatures = numOfFeatures + 1  # remember to include the intercept...
        self.alphaValue = alphaValue
        self.iterationValue = iterationValue
        self.weights = np.zeros(self.numOfFeatures)

    def predict(self, data):  # given alphaValue row, what is the predicted value
        return np.dot(data, self.weights)

    def fit(self, data, expectedValues):
        e = 0.000001  # minimum change needed to declare convergence
        for i in range(self.iterationValue):
            v = self.weights.copy()
            self.gradient_descent(data, expectedValues)
            total_adj = sum([abs(wi - vi) for wi, vi in zip(self.weights, v)])
            if total_adj < e:
                break
        self.out.write("%0.4f, %d, %0.4f, %0.4f, %0.4f\n" % (
        self.alphaValue, self.iterationValue, self.weights[0], self.weights[1], self.weights[2]))

    def gradient_descent(self, X: np.ndarray, Y: np.ndarray):
        weights = self.weights
        predict = self.predict  # abbreviations
        n = weights.shape[0]
        np.seterr(all='ignore')

        for i, x_i in enumerate(X):  # for each x_i in X (the training data set)
            for j in range(n):  # for each feature in the test data instance, x_i,j
                try:
                    weights[j] -= self.adjust_weights(Y[i], predict(x_i), x_i[j], self.alphaValue)
                except Exception as e:
                    print(e)
                    pass

    @staticmethod
    def adjust_weights(y, hx, xij, a):
        # get a proportional fraction of the feature and remove from the corresponding weight
        try:
            return a * xij * (hx - y)
            # return (hx - y)/(a*xij)
        except OverflowError:
            return 0.0

def import_and_scale(input_file, with_bias_column=True):
    raw_data = np.loadtxt(input_file, delimiter=',')
    scale_data = True

    age = raw_data[:, [0]]
    weight = raw_data[:, [1]]
    heights = raw_data[:, [2]].flatten()
    if scale_data:
        age = scale(raw_data[:, [0]])
        weight = scale(raw_data[:, [1]])
        heights = scale(raw_data[:, [2]]).flatten()

    if with_bias_column:
        rows = raw_data.shape[0]
        intercept_column = np.ones(rows)
        intercept_column.shape = (rows, 1)
        data = np.hstack((intercept_column, age, weight))
    else:
        data = np.hstack((age, weight))
    return data, heights


def main():
    data, heights = import_and_scale(sys.argv[1])
    of = open(sys.argv[2],'w')
    for iterationValue, alphaValue in [(100, 0.001), (100, 0.005), (100, 0.01),
                              (100, 0.05), (100, 0.1), (100, 0.5),
                              (100, 1), (100, 5), (100, 10)]:
        lr = LinearRegressor(iterationValue=iterationValue, alphaValue=alphaValue, of=of)
        lr.fit(data, heights)
    of.close()


if __name__ == "__main__":
    main()
