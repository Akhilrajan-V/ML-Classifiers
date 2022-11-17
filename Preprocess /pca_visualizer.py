import numpy as np
import cv2 as cv
import sys
'''
sys.path is a list of absolute path strings (DO NOT GIVE RELATIVE PATH)
'''

data_path = "/home/akhil/PycharmProjects/pythonProject/Spr_Proj1/Code/"
sys.path.append(data_path)

from Code.preprocess import train_data, test_data
import Code.PCA as PCA
import Code.MDA as MDA


# LOADING DATA
def load_data():
    X = np.array([img[0] for img in train_data['train']])
    X_dim = X.shape
    img_h = X_dim[1]
    img_w = X_dim[2]
    Y = np.array([label[1] for label in train_data['train']])
    X_test = np.array([img[0] for img in test_data['test']])
    Y_test = np.array([label[1] for label in test_data['test']])
    return X, X_test, Y, Y_test, X_dim, img_h, img_w


def main():
    X, X_test, Y, Y_test, X_dim, img_h, img_w = load_data()
    print("Choose DATA Preprocess to Visualize \n 1.PCA    2.MDA")
    inp = int(input("SELECT 1 or 2  :"))
    if inp == 1:
        print("PCA will be applied to data to keep 99% of Variance\n")
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
        a = PCA.PCA()
        a.__int__(X, X_dim)
        X_pca, X_viz = a.transform(a.X)
        solo = int(input("Do You want to visualize a single image? \n If yes Enter 1, If you want to visualize all "
                         "images Enter any number:"))
        if solo == 1:
            label = int(input("Enter Label of Image Indexed from (0 to 799) :"))
            cv.imshow('PCA Dimension Reduced Reconstructed', X_viz[label][:][:])
            cv.imshow('train', train_data['train'][label][0])
            cv.waitKey(0)
        else:
            for label in range(np.shape(X)[0]):
                cv.imshow('PCA Dimension Reduced Reconstructed', X_viz[label][:][:])
                cv.imshow('train', train_data['train'][label][0])
                cv.waitKey(0)
    elif inp == 2:
        n = int(input("Enter Number of Components :"))
        a = MDA.MDA()
        a.compute_MDA(X, Y, X_test, n, img_h, img_w, show_img=True)
    else:
        print(":{ Wrong input\n")
        print("Try again\n")
        main()


if __name__ == "__main__":
    main()

