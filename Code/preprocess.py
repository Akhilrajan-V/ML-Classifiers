import random

import numpy
import scipy.io as sc
import cv2 as cv
import numpy as np

mat = sc.loadmat('/home/akhil/PycharmProjects/pythonProject/Spr_Proj1/Data/data.mat')
data = mat['face']

train_data = dict()
test_data = dict()
t_data = dict()


def test_data_split(train_data):

    train_data['train'] = [train_data['train'][x:x + 3] for x in range(0, len(train_data['train']), 3)]
    for t in range(0, len(train_data['train'])):
        r = random.randint(0, 2)
        temp = train_data['train'][t].pop(r)
        # print(temp)
        test_data.setdefault("test", []).append(temp)
    # print(len(test_data['test']))
    train_data['train'] = sum(train_data['train'], [])
    # train_data.clear()
    # train_data.setdefault("train", chunks)


def split_data(dataset):
    img_per_class = 3
    total_img = np.shape(dataset)[2]
    label = 1
    count = 1
    for img in range(0, total_img):
        train_data.setdefault("train", []).append([dataset[:, :, img], label])
        if (count-3) == 0:
            label += 1
            count = 0
        count += 1
    test_data_split(train_data)


split_data(data)
i = 2
# print((train_data['train'][i][1]))
# cv.imshow('train',train_data['train'][i][0])
# cv.imshow('face', data[:,:,3])
# print((test_data['test'][1][1]))
# cv.imshow('test',test_data['test'][1][0])
# cv.waitKey(0)
# print(len(train_data['train']))
