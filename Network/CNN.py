from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Add, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation

def cnn_layer(inputs, n_class, last_layer_activation='softmax'):
    """
    自定义的简单网络层架构
    :param inputs: tensor，输入的数据
    :param n_class: int，最终的分类数量
    :param last_layer_activation: string, 最后一层的激活函数
    :return: tensor, 输出最后一层产生的tenosr。
    """

    # 224 * 224 * 3 --> 112 * 112 * 32
    x1 = Conv2D(filters=32,
               kernel_size=(7, 7),
               strides=(2, 2),
               padding='same',
               name='Conv1',
               kernel_initializer='random_normal',
               use_bias=True,
               bias_initializer='random_normal')(inputs)
    x = BatchNormalization(axis=-1, name='BN1')(x1)
    x = Activation('relu', name='activation1')(x)

    # 112 * 112 * 64 --> 56 * 56 * 128
    xi = Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(2, 2),
               padding='same',
               name='Conv2',
               kernel_initializer='random_normal',
               use_bias=True,
               bias_initializer='random_normal')(x)
    x = BatchNormalization(axis=-1, name='BN2')(xi)
    x = Activation('relu', name='activation2')(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               name='Conv3',
               kernel_initializer='random_normal',
               use_bias=True,
               bias_initializer='random_normal')(x)
    x = BatchNormalization(axis=-1, name='BN3')(x)
    x = Activation('relu', name='activation3')(x)
    x = Add(name='add1')([x, xi])

    # 56 * 56 * 128 --> 18 * 18 * 256
    x = Conv2D(filters=128,
               kernel_size=(3, 3),
               strides=(3, 3),
               name='Conv4',
               padding='valid',
               kernel_initializer='random_normal',
               use_bias=True,
               bias_initializer='random_normal')(x)
    x = BatchNormalization(axis=-1, name='BN4')(x)
    x = Activation('relu', name='activation4')(x)

    # 18 * 18 * 256 --> 9 * 9 * 512
    x = Conv2D(filters=256,
               kernel_size=(3, 3),
               strides=(2, 2),
               name='Conv5',
               padding='same',
               kernel_initializer='random_normal',
               use_bias=True,
               bias_initializer='random_normal')(x)
    x1 = Conv2D(filters=256,
                kernel_size=(5, 5),
                strides=(12, 12),
                name='Conv1_1',
                padding='valid',
                kernel_initializer='random_normal',
                use_bias=True,
                bias_initializer='random_normal')(x1)
    x = Add()([x, x1])
    x = BatchNormalization(axis=-1, name='BN5')(x)
    x = Activation('relu', name='activation5')(x)

    # 9 * 9 * 512 --> 1 * 1 * 512
    x = AveragePooling2D(pool_size=(9, 9), name='AvPo1')(x)

    # 1 * 1 * 512 --> 512
    x = Flatten(name='Flatten1')(x)

    # 512 --> 64
    x = Dense(64, name='Dense1')(x)
    x = BatchNormalization(axis=-1, name='BN6')(x)
    x - Activation('relu', name='activation6')(x)

    # 64 --> n_class
    x = Dense(n_class, name='Dense2')(x)
    x = Activation(last_layer_activation, name='out')(x)

    return x