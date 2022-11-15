import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance


class KNN:

    def __int__(self, k=3):
        self.K = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        X_test = X
        predictions = np.zeros(np.shape(X_test)[0])
        for i in range(np.shape(X_test)[0]):
            predictions[i] = self._predict(X_test[i][:], self.K)
        return predictions

    def _predict(self, x, k):
        distances = np.zeros(np.shape(self.X_train)[0])
        for i in range(self.X_train.shape[0]):
            distances[i] = euclidean_distance(x, self.X_train[i][:])
        k_indexes = np.argsort(distances)
        k_indexes = k_indexes[:k]
        k_labels = [self.Y_train[s] for s in k_indexes]
        most_k_label = Counter(k_labels).most_common(1)

        return most_k_label[0][0]

