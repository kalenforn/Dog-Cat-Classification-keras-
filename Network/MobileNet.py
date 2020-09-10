from keras.layers import Add, Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, AveragePooling2D, MaxPooling2D

from keras import backend as K

class MobileNet:
    def __init__(self, input_shape, n_class):
        """
        初始化类

        :param input_shape:输入图片的尺寸（w，h，c）
        :param n_class: 最终的分类数量
        """

        self.input_shape = input_shape
        self.n_class = n_class

    def _bottleneck(self, inputs, filters, strides=(1, 1), t=6, kernel=(3, 3), name='conv'):
        """
        bottleneck 子层的实现

        :param inputs: 输入的tensor
        :param filters: 输出的维度
        :param kernel: DW卷积核大小
        :param t:初次卷积的channel放大倍数
        :param strides: 卷积步数
        :param name:该层卷积的名称起始

        :return: 单层BN的tensor
        """

        inputs = BatchNormalization(axis=-1)(inputs)

        # 判断输入的通道数，channel_first\channel_last模式不同在于把channel的数量放在了第一个维度还是最后一个维度
        axis_channel = 1 if K.image_data_format() == 'channel_first' else -1
        input_shape = K.int_shape(inputs)
        #print(input_shape)
        channel = input_shape[axis_channel] * t
        add = strides[0] == 1 and strides[1] == 1 and filters == input_shape[axis_channel]

        x = Conv2D(channel, (1, 1), strides=(1, 1), padding='same', name=name+'_1')(inputs)
        x = Activation('relu')(x)
        x = DepthwiseConv2D(kernel, strides=strides, padding='same', name=name+'_2')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name=name+'_3')(x)

        if add:
            x = Add()([x, inputs])

        return x

    def build(self, _input, last_layer_activation='softmax'):
        """
        构建mobilenet V2网络架构

        :param _input: tensor, 输入的input
        :param last_layer_activation: string, 最后一层的激活函数

        :return: tensor, 输出最后一层产生的tenosr。
        """

        # batch_size * 3 * 224 * 224 --> batch_size * 32 * 112 * 112  Conv2d 1X1
        x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv1_1')(_input)
        x = Activation('relu')(x)

        # batch_size * 32 * 112 * 112 --> batch_size * 16 * 112 * 112 n=1
        x = self._bottleneck(inputs=x, filters=16, t=1, strides=(1, 1), name='conv2')

        # batch_size * 16 * 112 * 112 --> batch_size * 24 * 56 * 56 n=2
        x = self._bottleneck(inputs=x, filters=24, strides=(2, 2), name='conv3')
        x = self._bottleneck(inputs=x, filters=24, strides=(1, 1), name='conv4')

        # batch_size * 24 * 56 * 56 --> batch_size * 32 * 28 * 28 n=3
        x = self._bottleneck(inputs=x, filters=32, strides=(2, 2), name='conv5')
        for i in range(2):
            x = self._bottleneck(inputs=x, filters=32, strides=(1, 1), name='conv'+str(i+6))

        # batch_size * 32 * 28 * 28 --> batch_size * 64 * 14 * 14 n=4
        x = self._bottleneck(inputs=x, filters=64, strides=(2, 2), name='conv8')
        for i in range(3):
            x = self._bottleneck(inputs=x, filters=32, strides=(1, 1), name='conv'+str(i+9))

        # batch_size * 64 * 14 * 14 --> batch_size * 96 * 14 * 14 n=3
        for i in range(3):
            x = self._bottleneck(inputs=x, filters=96, strides=(1, 1), name='conv'+str(i+12))

        # batch_size * 96 * 14 * 14 --> batch_size * 160 * 7 * 7 n=3
        x = self._bottleneck(inputs=x, filters=160, strides=(2, 2), name='conv15')
        for i in range(2):
            x = self._bottleneck(inputs=x, filters=96, strides=(1, 1), name='conv'+str(i+16))

        # batch_size * 160 * 7 * 7 --> batch_size * 320 * 7 * 7 n=1
        x = self._bottleneck(inputs=x, filters=320, strides=(1, 1), name='conv18')

        # batch_size * 320 * 7 * 7 --> batch_size * 1280 * 7 * 7  Conv2d 1X1
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(1280, (1, 1), strides=(1, 1), padding='same', name='conv19')(x)
        x = Activation('relu')(x)
        #x = Dropout(name='dropoty')(x)

        # batch_size * 1280 * 7 * 7 --> batch_size * 1280 * 1 * 1  AvgPooling 7X7
        x = AveragePooling2D(pool_size=(7, 7))(x)
        #x = MaxPooling2D(pool_size=(7, 7))(x)

        # batch_size * 1280 * 1 * 1 --> n_class  Conv2d 1X1 --> n_class
        #output = Conv2D(self.n_class, (1, 1), padding='same', name='conv20')(x)
        # 此处选用sigmoid而不是softmax是因为sigmoid适合二元分类，dog-cat分类为二元分类
        #output = Activation('tanh', name='last_layer')(output)
        # 到此为止是由论文复述出来的MobileNet V2网络，因为项目中是需要将label做成
        # （？，2）形式的维度结果故在此后面加一个Flatten再做一个全连接使得模型特征值与标签相对应

        # batch_size * 1 * 1 * n_class --> batch_size * n_class  Flatten 拉平
        output = Flatten(name='flatten')(x)

        # 全连接层重新选择，主要是为了最后的softmax或者sigmoid
        #output = BatchNormalization(axis=-1)(output)
        #output = Dense(640, name='dense1', activation='relu')(output)
        #output = BatchNormalization(axis=-1)(output)
        #output = Dense(320, name='dense2', activation='relu')(output)
        # output = Dense(64, name='dense3', activation='relu')(output)
        output = Dense(self.n_class, name='out', activation=last_layer_activation)(output)

        return output
