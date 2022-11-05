import scipy.io as sc
import cv2 as cv
mat = sc.loadmat('./Data/data.mat')
data = mat['face']
cv.imshow('face', data[:,:,2])
cv.waitKey(0)
