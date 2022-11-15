import numpy as np
import scipy.io as sio
import random

import PCA
from bayes import Bayes
from knn import KNN

def load_POSE(filename):
    data = sio.loadmat(filename)
    dat = data['pose']
    train_imgs = []
    test_imgs = []
    train_labels = []
    test_labels = []
    for subj in range(0, dat.shape[3]):
        r = random.randint(0, 12)
        for p in range(0,dat.shape[2]):
            img = dat[:, :, p, subj]
            if p == r:
                test_imgs.append(img)
                test_labels.append(subj)
            else:
                train_imgs.append(img)
                train_labels.append(subj)

    train_imgs = np.array(train_imgs)
    test_imgs = np.array(test_imgs)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_imgs, test_imgs, train_labels, test_labels


X, X_test, Y, Y_test = load_POSE('/home/akhil/PycharmProjects/pythonProject/Spr_Proj1/Data/pose.mat')
Og_dim = (np.shape(X))

X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
print(np.shape(X))

# PCA
a = PCA.PCA()
a.__int__(X, Og_dim)

X_pca, X_viz = a.transform(a.X)
print(X_pca.shape)
project_mat = a.projection_matrix
X_test_pca = np.dot(X_test, project_mat)
print(X_test_pca.shape)

# Bayes
bayes = Bayes(X_pca, Y)
bayes.fit(X_pca, Y)
bayes_y_predicted = bayes.predict(X_test_pca)


def accuracy(y, y_pred):
    correct = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            correct += 1
    acc = (correct/len(y_pred)) * 100
    return acc


print("Bayes Classification accuracy = ", accuracy(Y_test, bayes_y_predicted))

