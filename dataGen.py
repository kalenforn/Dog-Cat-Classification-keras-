from config import IMG_SIZE, BATCH_SIZE

import numpy as np

import os
import cv2

class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n=[]
        if(self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index : self.index + batchsize]
            self.index = self.index + batchsize

        return r_n

# 生成数据集的迭代器
"""
def genData(path):
    '''
    Generator data for train or test.

    :param path: Image dir path,this is a dir path

    :return: yield object.
    '''

    filename = os.listdir(path)
    #print(len(filename))
    data = np.zeros((BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 3), np.float32)

    r_n = random_uniform_num(len(filename))
    filename = np.array(filename)
    label = None

    while 1:
        if label is not None:
            del label

        label = np.zeros(BATCH_SIZE, np.int32)
        shufflefile = filename[r_n.get(BATCH_SIZE)]
        for i, j in enumerate(shufflefile):
            if "cat" in j:
                label[i] = 1
            else:
                label[i] = 0

            img = cv2.imread(os.path.join(path, j))
            #print(img.shape)
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
            img = np.array(img, 'f') / 255.0
            #print(img.shape)

            data[i] = np.expand_dims(img, axis=0)

        #print(data.shape)
        _input = {"the_input": data}
        output = {"the_output": label}
        # print(label)
        yield (_input, output)

"""
def genData(path):
    '''
    Generator data for train or test.

    :param path: Image dir path,this is a dir path

    :return: yield object.
    '''

    filename = os.listdir(path)
    #print(len(filename))
    r_n = random_uniform_num(len(filename))
    filename = np.array(filename)
    label = None
    data = None

    while 1:
        if not label is None:
            del label
        if not data is None:
            del data

        label = np.zeros((BATCH_SIZE, 2), np.float32)
        ######################################################
        # data不刷新为什么会导致训练出错？
        ######################################################
        data = np.zeros((BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 3), np.float32)
        shufflefile = filename[r_n.get(BATCH_SIZE)]
        for i, j in enumerate(shufflefile):
            if "cat" in j:
                label[i][0] = 1.0
            else:
                label[i][1] = 1.0

            img = cv2.imread(os.path.join(path, j))
            #print(img.shape)
            img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
            img = np.array(img, 'f') / 255.0
            #print(img.shape)

            data[i] = np.expand_dims(img, axis=0)

        #print(data.shape)
        # label = np.ndarray(label)
        _input = {"the_input": data}
        output = {"out": label}
        # print(label)
        yield (_input, output)
