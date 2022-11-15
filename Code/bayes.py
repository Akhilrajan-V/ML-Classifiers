import numpy as np


class Bayes:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n_samples, self.n_features = self.X.shape
        self._classes = np.unique(Y)
        self.n_classes = len(self._classes)

        self._mean = None
        self._cov = None
        self._priors = None

    def fit(self, X, Y):
        # calculate mean, var. prior
        self._mean = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        self._cov = []
        self._priors = np.zeros(self.n_classes, dtype=np.float64)

        for i, cls in enumerate(self._classes):
            X_c = X[Y == cls]
            self._mean[i, :] = np.mean(X_c, axis=0)
            self._cov.append(np.cov(X_c.T))
            self._priors[i] = (np.shape(X_c)[0]) / self.n_classes
        # self.log(self._cov)

    def log(self, x, X):
        print("Test Image: {}/{}".format(x, X))

    def precdet(self, sigma, threshold):
        deter = np.linalg.det(sigma)
        if deter < 1e-06:
            w, v = np.linalg.eig(sigma)
            deter = np.product(np.real(w[w > threshold]))
            precision = np.linalg.pinv(sigma)
        else:
            precision = np.linalg.inv(sigma)
        return precision, deter

    def _pdf(self, x, index):
        mean = self._mean[index][:]
        cov = self._cov[index]
        cov += 0.0001*np.identity(np.shape(cov)[0])
        temp1 = (np.subtract(x, mean)).T
        cov_inv, cov_det = self.precdet(cov, 0.5)
        temp = np.dot(temp1, cov_inv)
        temp2 = np.dot(temp, (x - mean))
        numerator = np.exp((-1/2)*temp2)
        denominator = np.sqrt(2 * np.pi) * np.sqrt(cov_det)
        # print("numera", np.shape(cov_inv))
        # print("edn", denominator)
        return numerator/denominator

    def predict(self, X):
        y_predicted = np.zeros(np.shape(X)[0])
        for i in range(np.shape(X)[0]):
            y_predicted[i] = (np.array(self._predict(X[i][:])))
            self.log(i+1, np.shape(X)[0])
        return y_predicted

    def _predict(self, x):
        posteriors = []

        # Calculate the Posterior for each class
        for i, cls in enumerate(self._classes):
            prior = np.log(self._priors[i])
            posterior = np.log(self._pdf(x, i))
            posterior = posterior + prior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]




