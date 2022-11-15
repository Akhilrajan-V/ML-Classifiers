import numpy as np
import matplotlib.pyplot as plt
from preprocess import train_data, test_data
# from preprocess import X_train_nvs, Y_train_nvs, X_test_nvs, Y_test_nvs
# from preprocess import X_val_nvs, Y_val_nvs
from preprocess import n_v_ex_data_gen
import PCA
import MDA
from bayes import Bayes
from knn import KNN
from svm import SVM

"""
# call functions and methods
"""

Data_PATH = '/home/akhil/PycharmProjects/pythonProject/Spr_Proj1/Data/data.mat'


print("ALL SELECTION OPTIONS ARE INT INPUTS")
print("_____________________________________")
print("1.Face Label Recognition     2.Neutral vs Expression Detection")
task = int(input("Enter 1 or 2 :"))
print("")
print("Select Dimensionality reduction algorithm")
print("1.PCA    2.MDA")
red = int(input("Enter 1 or 2 :"))
print("")

if task==1:
    print("Select a Classifier")
    print("1.Bayes    2.KNN")
    classifier = int(input("Please Select a Classifier by entering the corresponding number: "))
    print("")
elif task==2:
    print("Select a Classifier")
    print("NOTE: To Run BOOST SVM open boosted_svm and run it directly")
    print("")
    print("1.Bayes   2.KNN    3.SVM ")
    classifier = int(input("Please Select a Classifier by entering the corresponding number :"))


def accuracy(y, y_pred):
    correct = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            correct += 1
    acc = (correct/len(y_pred)) * 100
    return acc


def cross_validate_svm(X, Y, X_val, Y_val, kernel):

    if kernel=="polynomial":
        accu = []
        svm_linear = SVM()
        svm_linear.__int__()
        R = np.arange(0, 6, 0.5)
        for r in range(len(R)):
            deg = R[r]
            w, b, losses = svm_linear.fit(X, Y, X_val, Y_val, kernel=kernel, r=deg)
            svm_prediction = svm_linear.predict(X_val)
            accu.append(accuracy(Y_val, svm_prediction))
        max_value = max(accu)
        index = accu.index(max_value)
        print("Optimal value of r: ", R[index])
        return R[index]

    elif kernel=="RBF":
        accu = []
        svm_linear = SVM()
        svm_linear.__int__()
        Sigma = np.arange(1, 12, 1)
        for s in range(len(Sigma)):
            sigma = Sigma[s]
            w, b, losses = svm_linear.fit(X, Y, X_val, Y_val, kernel=kernel, sigma=sigma)
            svm_prediction = svm_linear.predict(X_val)
            accu.append(accuracy(Y_val, svm_prediction))
        max_value = max(accu)
        index = accu.index(max_value)
        print("Optimal value of Sigma: ", Sigma[index])
        return Sigma[index]

    # print("SVM with {} Kernel -- Accuracy={}%".format(kernel, accuracy(Y_test_, svm_prediction)))


def choose_class(c, X_train, Y_train, X_test, Y_test_):
    if c==1:
        # Bayes
        bayes = Bayes(X_train, Y_train)
        bayes.fit(X_train, Y_train)
        predicted_y = bayes.predict(X_test)
        accu = accuracy(Y_test_, predicted_y)
        print("Accuracy of Bayes Classifier {}%".format(accu))

    elif c==2:
        # KNN
        k = int(input("Enter K Component value: "))
        knn = KNN()
        knn.__int__(k=k)
        knn.fit(X_train, Y_train)
        knn_y_predicted = knn.predict(X_test)
        # print("Predicted y :", y_predicted)
        accu = accuracy(Y_test_, knn_y_predicted)
        print("Accuracy of KNN Classifier {}%".format(accu))

    elif c==3:
        # SVM
        print("Choose Kernel")
        print("Default  is linear SVM")
        print("1.Polynomial Kernel     2.RBF Kernel       3.Default")
        k = int(input("Enter anyone of the options by entering the corresponding number :"))
        if k==1:
            kernel="polynomial"
            svm_p = SVM()
            svm_p.__int__()
            # print(np.shape(X_val_nvs))
            # X_val_nvs = X_val_nvs.reshape((X_val_nvs.shape[0], X_val_nvs.shape[1] * X_val_nvs.shape[2]))
            r = cross_validate_svm(X_train, Y_train, X_val_nvs, Y_val_nvs, kernel)
            w, b, losses = svm_p.fit(X_train, Y_train, X_val_nvs, Y_val_nvs, kernel=kernel, r=r)
            svm_prediction = svm_p.predict(X_test)
            print("SVM with {} Kernel -- Accuracy={}%".format(kernel, accuracy(Y_test_, svm_prediction)))
            plt.title("training Loss")
            plt.plot(losses)
            plt.xlabel("Iterations")
            plt.ylabel("Error")
            plt.show()

        elif k==2:
            kernel="RBF"
            svm_p = SVM()
            svm_p.__int__()
            # print(np.shape(X_val_nvs))
            # X_val_nvs = X_val_nvs.reshape((X_val_nvs.shape[0], X_val_nvs.shape[1] * X_val_nvs.shape[2]))
            sigma = cross_validate_svm(X_train, Y_train, X_val_nvs, Y_val_nvs, kernel)
            w, b, losses = svm_p.fit(X_train, Y_train, X_val_nvs, Y_val_nvs, kernel=kernel, sigma=sigma)
            svm_prediction = svm_p.predict(X_test)
            print("SVM with {} Kernel -- Accuracy={}%".format(kernel, accuracy(Y_test_, svm_prediction)))
            plt.title("training Loss")
            plt.plot(losses)
            plt.xlabel("Iterations")
            plt.ylabel("Error")
            plt.show()

        else:
            print("Invalid defaulting")
            kernel=None


if task == 1:

    X = np.array([img[0] for img in train_data['train']])
    X_dim = X.shape
    img_h = X_dim[1]
    img_w = X_dim[2]
    Y = np.array([label[1] for label in train_data['train']])
    X_test = np.array([img[0] for img in test_data['test']])
    Y_test = np.array([label[1] for label in test_data['test']])

    if red==1:

        # PCA
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

        a = PCA.PCA()
        a.__int__(X, X_dim)

        X_pca, X_viz = a.transform(a.X)
        print(X_pca.shape)
        project_mat = a.projection_matrix
        X_test_pca = np.dot(X_test, project_mat)
        # print(X_test_pca[0:50][:].shape)
        choose_class(classifier, X_pca, Y, X_test_pca, Y_test)

    elif red ==2:
        md = MDA.MDA()
        X_mda, X_test_mda = md.compute_MDA(X, Y, X_test, 200, img_h, img_w, show_img=False)
        # X_mda = X_mda.reshape((X_mda.shape[0], X_mda.shape[1] * X_mda.shape[2]))
        # X_test_mda = X_test_mda.reshape((X_test_mda.shape[0], X_test_mda.shape[1] * X_test_mda.shape[2]))
        choose_class(classifier, X_mda, Y, X_test_mda, Y_test)


if task == 2:
    X_train_nvs, X_test_nvs, X_val_nvs, Y_train_nvs, Y_test_nvs, Y_val_nvs = n_v_ex_data_gen(Data_PATH)

    X_nvs_dim = np.shape(X_train_nvs)
    img_h = X_nvs_dim[1]
    img_w = X_nvs_dim[2]

    if red == 1:
        # PCA
        X_train_nvs = X_train_nvs.reshape((X_train_nvs.shape[0], X_train_nvs.shape[1] * X_train_nvs.shape[2]))
        X_test_nvs = X_test_nvs.reshape((X_test_nvs.shape[0], X_test_nvs.shape[1] * X_test_nvs.shape[2]))
        X_val_nvs = X_val_nvs.reshape((X_val_nvs.shape[0], X_val_nvs.shape[1] * X_val_nvs.shape[2]))
        b = PCA.PCA()
        b.__int__(X_train_nvs, X_nvs_dim)
        X_pca_nvs, X_viz_nvs = b.transform(b.X)
        project_mat_nvs = b.projection_matrix
        X_test_pca_nvs = np.matmul(X_test_nvs, project_mat_nvs)
        X_val_nvs = np.matmul(X_val_nvs, project_mat_nvs)
        choose_class(classifier, X_pca_nvs, Y_train_nvs, X_test_pca_nvs, Y_test_nvs)


    elif red==2:

        if classifier==3:
            md = MDA.MDA()
            X_mda, X_test_mda, X_val_nvs = md.compute_MDA_svm(X_train_nvs, Y_train_nvs, X_test_nvs, X_val_nvs, 200, img_h, img_w, show_img=False)
            choose_class(classifier, X_mda, Y_train_nvs, X_test_mda, Y_test_nvs)
        else:
            md = MDA.MDA()
            X_mda, X_test_mda = md.compute_MDA(X_train_nvs, Y_train_nvs, X_test_nvs, 200, img_h, img_w, show_img=False)
            # X_mda = X_mda.reshape((X_mda.shape[0], X_mda.shape[1] * X_mda.shape[2]))
            # X_test_mda = X_test_mda.reshape((X_test_mda.shape[0], X_test_mda.shape[1] * X_test_mda.shape[2]))
            choose_class(classifier, X_mda, Y_train_nvs, X_test_mda, Y_test_nvs)



