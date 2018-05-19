import numpy as np
from train import Train

class Predict:
    def __init__(self, X_test, y_test):
        self.train = Train()
        self.Xtest = X_test
        self.ytest = y_test
        self.num_test = self.Xtest.shape[0]
        self.y_pred = np.zeros(self.num_test, dtype=self.train.y.dtype)

    def predict(self):
        for i in range(self.num_test):
            distances = np.sum(np.abs(self.Xtest - self.train.X[i,:]), axis=1)
            min_idx = np.argmin(distances)
            self.y_pred[i] = self.train.y[min_idx]
        return  self.y_pred
