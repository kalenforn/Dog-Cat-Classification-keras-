from keras.layers import Add, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D

def resnet_block(inputs,
              num_out,
              kernel,
              strdies=(1, 1),
              padding='same',
              use_bias=False,
              name='conv1',
              io_concate=False):
    """
    resnet block
    :param inputs: a tenosr or an image data
    :param num_out: int, output channel number
    :param kernel: tuple, filter size
    :param strdies: tuple, strides size
    :param padding: boolean, padding or not
    :param use_bias: boolean, weather use bias
    :param name: string, layer's name
    :param io_concate: boolean, does the input channel same as the output channel in this layer

    :return: a tensor, product a single layer of ResNet
    """

    conv1 = Conv2D(num_out, kernel_size=kernel, strides=strdies,
                   padding=padding, use_bias=use_bias, name=name+'_1',
                   activation='relu', kernel_initializer='random_normal',
                   bias_initializer='random_normal')(inputs)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv2 = Conv2D(num_out, kernel_size=kernel, strides=(1, 1),
                   padding=padding, activation='relu', use_bias=use_bias,
                   name=name+'_2', kernel_initializer='random_normal',
                   bias_initializer='random_normal')(conv1)
    conv2 = BatchNormalization(axis=-1)(conv2)

    if io_concate:
        inputs = Conv2D(num_out, kernel_size=(1, 1), strides=strdies,
                        padding=padding, activation='relu', use_bias=use_bias,
                        name=name+'_0', kernel_initializer='random_normal',
                        bias_initializer='random_normal')(inputs)

    return Add()([conv2, inputs])

def resnet34(inputs, n_class, last_layer_activation='softmax'):
    """
    ResNet34层网络架构
    :param inputs: tensor，输入的数据
    :param n_class: int，最终的分类数量
    :param last_layer_activation: string，最后一层的激活函数
    :return: tensor, 输出最后一层产生的tenosr。
    """

    # 224 * 224 * 3 --> 112 * 112 * 64
    x = Conv2D(64, kernel_size=(7, 7),
               strides=(2, 2),
               padding='same',
               activation='relu',
               name='conv1',
               use_bias=False,
               kernel_initializer='random_normal',
               bias_initializer='random_normal')(inputs)
    x = BatchNormalization(axis=-1)(x)

    # 112 * 112 * 64 --> 56 * 56 * 64  MaxPooling
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='MaxPool_1')(x)
    # 112 * 112 * 64 --> 56 * 56 * 64
    x = resnet_block(inputs=x, num_out=64, kernel=(3, 3), use_bias=True, name='conv2')
    x = resnet_block(inputs=x, num_out=64, kernel=(3, 3), use_bias=True, name='conv3')
    x = resnet_block(inputs=x, num_out=64, kernel=(3, 3), use_bias=True, name='conv4')

    # 56 * 56 * 64 --> 28 * 28 * 128
    x = resnet_block(inputs=x, num_out=128, kernel=(3, 3), strdies=(2, 2),
                     use_bias=True, io_concate=True, name='conv5')
    x = resnet_block(inputs=x, num_out=128, kernel=(3, 3), use_bias=True, name='conv6')
    x = resnet_block(inputs=x, num_out=128, kernel=(3, 3), use_bias=True, name='conv7')
    x = resnet_block(inputs=x, num_out=128, kernel=(3, 3), use_bias=True, name='conv8')

    # 28 * 28 * 128 --> 14 * 14 * 256
    x = resnet_block(inputs=x, num_out=256, kernel=(3, 3), strdies=(2, 2),
                     use_bias=True, io_concate=True, name='conv9')
    x = resnet_block(inputs=x, num_out=256, kernel=(3, 3), use_bias=True, name='conv10')
    x = resnet_block(inputs=x, num_out=256, kernel=(3, 3), use_bias=True, name='conv11')
    x = resnet_block(inputs=x, num_out=256, kernel=(3, 3), use_bias=True, name='conv12')
    x = resnet_block(inputs=x, num_out=256, kernel=(3, 3), use_bias=True, name='conv13')
    x = resnet_block(inputs=x, num_out=256, kernel=(3, 3), use_bias=True, name='conv14')

    # 14 * 14 * 256 --> 7 * 7 * 512
    x = resnet_block(inputs=x, num_out=512, kernel=(3, 3), strdies=(2, 2),
                     use_bias=True, io_concate=True, name='conv15')
    x = resnet_block(inputs=x, num_out=512, kernel=(3, 3), use_bias=True, name='conv16')
    x = resnet_block(inputs=x, num_out=512, kernel=(3, 3), use_bias=True, name='conv17')

    # 7 * 7 * 512 --> 1 * 1 * 512
    x = AveragePooling2D(pool_size=(7, 7), name='AveragePooling1')(x)

    # 1 * 1 * 512 --> 512
    x = Flatten(name='flatten')(x)

    # 512 --> n_class
    x = Dense(n_class, activation=last_layer_activation, name='out')(x)

    return x



