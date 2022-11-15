import numpy as np


class SVM:

    def __int__(self, c=1.0, learning_rate=0.001):
        self.C = c
        self.lr = learning_rate
        self.w = 0
        self.b = 0

    def hinge_loss(self, w, b, X, Y):

        loss = 0
        # Regularize Term
        reg = 0.5 * (w*w)

        # Error Term
        for i in range(X.shape[0]):
            opt = Y[i] * ((np.dot(w, X[i][:]))+b)
            loss = reg + self.C * max(0, 1-opt)
            # print("loss", loss)
        return loss[0][0]

    def polynomial_kernel(self, X1, X2, r):
        X_transformed = np.power(((np.matmul(X1, np.transpose(X2)))+np.ones((np.shape(X1)[0], np.shape(X2)[0]))), r)
        return X_transformed

    def rbf_kernel(self, X1, X2, sigma):
        X_transformed = np.zeros((np.shape(X1)[0], np.shape(X2)[0]))
        for k in range(np.shape(X1)[0]):
            for m in range(np.shape(X2)[0]):
                temp = ((np.linalg.norm(X1[k] - X2[m]))**2)/sigma**2
                X_transformed[k][m] = np.exp(-temp)
        return X_transformed

    def fit(self, X, Y, X_val, Y_val, epochs=1000, kernel=None, r=None, sigma=None):
        self.kernel = kernel
        self.initial = X.copy()
        self.X_val = X_val
        self.Y_val = Y_val
        self.r = r
        self.sigma = sigma

        if self.kernel == "polynomial":
            # self.r = 5
            X = self.polynomial_kernel(X, X, self.r)

        if self.kernel == "RBF":
            # self.sigma = 10
            X = self.rbf_kernel(X, X, self.sigma)

        # y = np.where(Y <= 0, -1, 1)
        self.epochs = epochs
        n_samples, n_features = np.shape(X)
        w = np.zeros((1, n_features))
        b = 0
        losses=[]

        for i in range(self.epochs):

            loss = self.hinge_loss(w, b, X, Y)
            losses.append(loss)

            # Gradient Descent
            grad_w = 0
            grad_b = 0
            for j in range(n_samples):

                out = Y[j] * (np.matmul(w, np.transpose(X[j])) + b)

                if out > 1:
                    grad_w+=0
                    grad_b+=0
                else:
                    grad_w += self.C * Y[j] * X[j]
                    grad_b += self.C * Y[j]

            # Weight updates
            w = w - self.lr * w + self.lr * grad_w
            b = b + self.lr * grad_b

        # Returning Optimized weights
        self.w = w
        self.b = b

        return self.w, self.b, losses


    def predict(self, X_test, single_data=False):
        if single_data:

            if self.kernel == "polynomial":
                X_test = self.polynomial_kernel(X_test, self.initial, self.r)

            if self.kernel == "RBF":
                X_test = self.rbf_kernel(X_test, self.initial, self.sigma)

            prediction = np.dot(X_test, self.w[0]) + self.b
            return np.sign(prediction)
        else:

            if self.kernel == "polynomial":
                X_test = self.polynomial_kernel(X_test, self.initial, self.r)

            if self.kernel == "RBF":
                X_test = self.rbf_kernel(X_test, self.initial, self.sigma)

            predictions = np.zeros(np.shape(X_test)[0])
            for i in range(np.shape(X_test)[0]):
                predictions[i] = np.sign(np.dot(self.w, X_test[i])+self.b)
            return predictions
