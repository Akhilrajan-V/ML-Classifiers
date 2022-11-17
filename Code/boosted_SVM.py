import numpy as np
from preprocess import n_v_ex_data_gen
from svm import SVM
import PCA
import matplotlib.pyplot as plt

data_path = '/home/akhil/PycharmProjects/pythonProject/Spr_Proj1/Data/data.mat'
X_train_nvs, X_test_nvs, X_val_nvs, Y_train_nvs, Y_test_nvs, Y_val_nvs = n_v_ex_data_gen(data_path)


X_nvs_dim = np.shape(X_train_nvs)
X = X_train_nvs.reshape((X_train_nvs.shape[0], X_train_nvs.shape[1]*X_train_nvs.shape[2]))
X_test = X_test_nvs.reshape((X_test_nvs.shape[0], X_test_nvs.shape[1]*X_test_nvs.shape[2]))
X_val_nvs = X_val_nvs.reshape((X_val_nvs.shape[0], X_val_nvs.shape[1]*X_val_nvs.shape[2]))

# PCA
pc = PCA.PCA()
pc.__int__(X, X_nvs_dim)

X_pca_nvs, X_viz_nvs = pc.transform(pc.X)
project_mat_nvs = pc.projection_matrix
X_test_pca_nvs = np.matmul(X_test, project_mat_nvs)
X_pca_val = np.matmul(X_val_nvs, project_mat_nvs)


def accuracy(y, y_pred):
    correct = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            correct += 1
    acc = (correct/len(y_pred)) * 100
    return acc


a = SVM()
a.__int__(c=0.5) # 1 SVM
b = SVM()
b.__int__(c=0.9) # 2 SWM
c = SVM()
c.__int__(c=0.8) # 3 SVM
svm_ = [a, b, c]

acc =[]
svm_prediction = []
ws = []
bs = []
for i in range(0, 3):
    train_idx = np.arange(0, np.shape(X_train_nvs)[0], 1)
    np.random.shuffle(train_idx)
    index = train_idx[0:45]
    w, b, losses = svm_[i].fit(X_pca_nvs[index], Y_train_nvs[index], X_pca_val, Y_val_nvs)
    svm_prediction.append(svm_[i].predict(X_test_pca_nvs))
    acc.append(accuracy(Y_train_nvs[index], svm_prediction[i]))
    ws.append(w)
    bs.append(b)
ws = np.array(ws)
bs = np.array(bs)
print(ws.shape)
data_weights = np.ones(X_pca_nvs.shape[0])
a = []
count = 0
acc =[]
for k in range(0, np.shape(svm_prediction)[0]):
    eps = 0
    data_weights = data_weights / np.sum(data_weights)
    for i in range(X_pca_nvs.shape[0]):
        P_i = data_weights[i]
        pred_label = (svm_[k].predict(X_pca_nvs[i], single_data=True))
        if pred_label != Y_train_nvs[i]:
            eps += P_i
    a.append(0.5*np.log((1-eps)/eps))
    for i in range(X_pca_nvs.shape[0]):
        pred_label = np.sign(svm_[k].predict(X_pca_nvs[i], single_data=True))
        data_weights[i] = data_weights[i]*np.exp(-Y_train_nvs[i]*pred_label*a[k])
    count = 0
    for i in range(0, X_test_pca_nvs.shape[0]):
        pred_label = 0
        for q in range(0, len(a)):
            pred_label += a[q] * svm_[q].predict(X_test_pca_nvs[i], single_data=True)
        if np.sign(pred_label) == Y_test_nvs[i]:
            count += 1

    print("Accuracy of Boosted SVM after iteration {}: ".format(k), count / X_test_pca_nvs.shape[0])
    acc.append(count / X_test_pca_nvs.shape[0])
itr = np.linspace(1, 3, num=3)

plt.plot(itr, acc)
plt.show()


count=0
for i in range(0, X_test_pca_nvs.shape[0]):
    pred_label = 0
    for k in range(0,len(svm_)):
        pred_label += a[k]*(svm_[k].predict(X_test_pca_nvs[i], single_data=True))
    if np.sign(pred_label) == Y_test_nvs[i]:
        count += 1

print("Accuracy of 1st SVM: ", acc[0])
print("Accuracy of 2nd SVM: ", acc[1])
print("Accuracy of 3rd SVM: ", acc[2])
print("Accuracy for AdaBoost: ", (count / X_test_pca_nvs.shape[0])*100)



