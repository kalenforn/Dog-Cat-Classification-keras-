from matplotlib import pyplot as plt

import os

def gen_picture(history, path):
    """
    生成训练结果的可视化图像
    :param history: dict,训练结果的返回值
    :param path: string,图片存储路径
    """

    val_acc = history['val_acc']
    val_loss = history['val_loss']
    loss = history['loss']
    acc = history['acc']

    plt.figure(1)
    plt.plot(val_acc, label='val_acc', color='r', marker='.')
    plt.plot(acc, linestyle='--', label='acc', color='b', marker='.')
    plt.title('Training-Accuracy Vs Testing-Accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(path, 'A.png'))

    plt.figure(2)
    plt.plot(loss, linestyle='--', label='loss', color='b', marker='.')
    plt.plot(val_loss, label='val_loss', color='r', marker='.')
    plt.title('Training-Loss Vs Testing-loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(path, 'L.png'))
    plt.show()
