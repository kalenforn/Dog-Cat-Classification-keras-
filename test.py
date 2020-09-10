from keras import Model
from keras.layers import Input
from config import IMG_SIZE, N_CALSS
from Network.CNN import cnn_layer
from Network.MobileNet import MobileNet
from Network.ResNet import resnet34

import numpy as np
import argparse as arg

import os
import shutil
import cv2

def getArgs():

    parser = arg.ArgumentParser(description="Get your testing arguments")
    parser.add_argument('--image-path', '-im',default='./0.jpg', type=str, help='path to the test image dir')
    parser.add_argument('--model-path', '-mp', default='./model/test.h5', type=str, help='path to your testing model')
    parser.add_argument('--model', '-m', default=0, type=int, help='select network to test,0: CNN, 1: MobileNet; 2: ResNet34')
    args = parser.parse_args()

    return args

def getImage(image_path):
    """
    Get image
    :param image_path: image path

    :return: numpy object, reading image.
    """

    img = cv2.imread(image_path)
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
    img = np.array(img, 'f') / 255.0
    img = np.reshape(img, (1, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    #print(img.shape)

    return img

def get_model(model_path, model):
    """
    获取网络模型
    :param model_path: 模型的保存路径
    :param model: 网络的模式选择：0: CNN,；1: MobileNet; 2: ResNet34
    :return: 获取的模型
    """

    if not os.path.exists(model_path):
        print("Model path does not exist!")
        exit(0)

    _input = Input(shape=IMG_SIZE, name='the_input')
    if model == 0:
        y_pred = cnn_layer(inputs=_input, n_class=N_CALSS)
    elif model == 1:
        network = MobileNet(IMG_SIZE, N_CALSS)
        y_pred = network.build(_input=_input)
    elif model == 2:
        y_pred = resnet34(inputs=_input, n_class=N_CALSS)
    else:
        raise(TypeError("The param model must in [0, 1, 2]!"))

    basemodel = Model(inputs=_input, outputs=y_pred)
    basemodel.load_weights(model_path)

    return basemodel

def predict(basemodel, image_path):
    """
    Predict the animal in the picture is dog or cat.
    :param image_path: Image local path
    :param model_path: Model local path
    :return: String object, the recognition of this image.
    """

    if not os.path.exists(image_path):
        print("Image does not exist!")
        exit(0)

    img = getImage(image_path)

    pred = basemodel.predict(img)
    pred = pred[0]
    #pred_l1 = Model(inputs=_input, outputs=basemodel.get_layer(name='AveragePooling1').output)
    #pred_l1 = pred_l1.predict(img)
    #print("################################\nL1:\n", pred_l1, "\n################################")
    print(pred)


    if pred[0] > pred[1] :
        result = (0, pred[0]) # cat
    elif pred[1] > pred[0]:
        result = (1, pred[1]) # dog
    else:
        result = None # unkonw

    return result


if __name__ == '__main__':
    args = getArgs()
    basesmodel = get_model(args.model_path, args.model)
    _tuple = predict(basesmodel, args.image_path)

    if not _tuple is None:
        result, pred = _tuple
        result = "cat" if result == 0 else "dog"
        pred = round(pred, 4) * 100
        print("This is a {}, \n".format(result) + "The probability is %.*f" % (2, pred) + '%')
    else:
        print("Nothing in this picture.")
"""
    file = os.listdir(args.image_path)

    if not (os.path.exists(os.path.join(args.image_path, 'dog')) and
                        os.path.exists(os.path.join(args.image_path, 'cat'))):
        os.mkdir(os.path.join(args.image_path, 'dog'))
        os.mkdir(os.path.join(args.image_path, 'cat'))
        print("Making dir:{} and {}".format(os.path.join(args.image_path, 'dog'), os.path.join(args.image_path, 'cat')))

    # filepath = [args.image_path + item for item in file]
    basesmodel = get_model(args.model_path)

    for i, j in enumerate(file):
        if not '.jpg' in j:
            #print("aaa")
            continue
        flag = predict(basesmodel, os.path.join(args.image_path, j))
        if flag == 0:
            shutil.move(os.path.join(args.image_path, j), os.path.join(args.image_path, 'cat'))
        elif flag == 1:
            shutil.move(os.path.join(args.image_path, j), os.path.join(args.image_path, 'dog'))
        else:
            pass
"""


