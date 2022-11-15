import random
import scipy.io as sc
import numpy as np

mat = sc.loadmat('/home/akhil/PycharmProjects/pythonProject/Spr_Proj1/Data/data.mat')
data = mat['face']

train_data = dict()
test_data = dict()


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
        if (count - 3) == 0:
            label += 1
            count = 0
        count += 1
    test_data_split(train_data)


split_data(data)

def n_v_ex_data_gen(filename):
    data = sc.loadmat(filename)
    dat = data['face']
    train_imgs = []
    test_imgs = []
    val_imgs = []
    train_labels = []
    test_labels = []
    val_labels = []
    for n in range(1, 151):
        imgn = dat[:, :, 3 * n - 3]
        imgf = dat[:, :, 3 * n - 2]

        train_imgs.append(imgn)
        train_labels.append(-1)
        train_imgs.append(imgf)
        train_labels.append(1)

    for n in range(151, 176):
        imgn = dat[:, :, 3 * n - 3]
        imgf = dat[:, :, 3 * n - 2]

        val_imgs.append(imgn)
        val_labels.append(-1)
        val_imgs.append(imgf)
        val_labels.append(1)

    for n in range(176, 201):
        imgn = dat[:, :, 3 * n - 3]
        imgf = dat[:, :, 3 * n - 2]

        test_imgs.append(imgn)
        test_labels.append(-1)
        test_imgs.append(imgf)
        test_labels.append(1)

    train_imgs = np.array(train_imgs)
    test_imgs = np.array(test_imgs)
    val_imgs = np.array(val_imgs)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    val_labels = np.array(val_labels)
    return train_imgs, test_imgs, val_imgs, train_labels, test_labels, val_labels


