from keras.layers import Input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import Adam, SGD, RMSprop
from Network.MobileNet import MobileNet
from Network.ResNet import resnet34
from Network.CNN import cnn_layer
from dataGen import genData
from config import LAST_LAYER_ACTIVATION, PRE_MODEL_PATH, BATCH_SIZE, EPOCH, N_CALSS, IMG_SIZE, FILE_PATH_TRAIN, FILE_PATH_TEST, SAVE_DIR

from keras import backend as K

import numpy as np

import os

# 适合网络层最后用softmax的loss计算
def sofrmaxLoss(y_true, y_pred):
    loss = K.abs(K.sum(y_true * K.log(y_pred)))
    return loss


def train(model):
    """
    选择训练方式

    :param model: 0,1,2。0用自己写的cnn来训练；1用MobileNetV2训练；2用ResNet34训练

    :return: 训练结果的历史记录
    """

    _input = Input(shape=IMG_SIZE, name='the_input')

    #y_pred = cnn_layer(inputs=_input, n_class=N_CALSS)
    if model == 0:
        y_pred = cnn_layer(inputs=_input, n_class=N_CALSS, last_layer_activation=LAST_LAYER_ACTIVATION)
    elif model == 1:
        network = MobileNet(IMG_SIZE, N_CALSS)
        y_pred = network.build(_input=_input, last_layer_activation=LAST_LAYER_ACTIVATION)
    elif model == 2:
        y_pred = resnet34(inputs=_input, n_class=N_CALSS, last_layer_activation=LAST_LAYER_ACTIVATION)
    else:
        raise(TypeError("The param 'model' must in [0, 1, 2]!"))

    model = Model(inputs=_input, outputs=y_pred)

    #opt = SGD(lr=0.001, momentum=0.5, decay=1e-6)
    #opt = Adam(lr=0.001, decay=1e-6)
    #opt = RMSprop(lr=0.0001, decay=1e-6)
    # 加载与训练模型
    if os.path.exists(PRE_MODEL_PATH):
        print("Loading model weights...")
        basemodel = Model(inputs=_input, outputs=y_pred)
        basemodel.summary()
        basemodel = basemodel.load_weights(PRE_MODEL_PATH)
        print("Done!")
    # 以下是可选的loss，其计算方法请查询官方文档。也可以自定义计算loss的方法。
    # binary_crossentropy适合输出标签为1个的、categorical_crossentropy、sparse_categorical_crossentropy、poisson、kl_divergence
    # mean_squared_error、mean_absolute_error、mean_absolute_percentage_error、mean_squared_logarithmic_error、cosine_similarity、huber、log_cosh
    # hinge、squared_hinge、categorical_hinge
    model.compile(loss=sofrmaxLoss, optimizer='adam', metrics=['accuracy'])

    """
    train_data = ImageDataGenerator(rescale=1. / 255)
    test_data = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_data.flow_from_directory(
        directory=FILE_PATH_TRAIN,
        target_size=(IMG_SIZE[0], IMG_SIZE[1]),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    test_generator = test_data.flow_from_directory(
        directory=FILE_PATH_TEST,
        target_size=(IMG_SIZE[0], IMG_SIZE[1]),
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    """
    train_generator = genData("D:/data/Dog-cat/train")
    test_generator = genData("D:/data/Dog-cat/test")
    save_dir = SAVE_DIR
    if not os.path.exists(save_dir):
        print("Making Dir:", save_dir)
        os.mkdir(save_dir)

    checkpoint = ModelCheckpoint(filepath=save_dir + '/test-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)
    lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(EPOCH)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    tensorboard = TensorBoard(log_dir=save_dir + '/logs', write_graph=True)

    print("---------------start training---------------")
    history = model.fit_generator(train_generator,
        steps_per_epoch=24000 // BATCH_SIZE,
        epochs=EPOCH,
        initial_epoch=0,
        validation_data=test_generator,
        validation_steps=1000 // BATCH_SIZE,
        callbacks=[checkpoint, changelr, tensorboard]
    )
    return history.history
    #output = Model(inputs=_input, outputs=model.get_layer(name='conv1_1').output)
    #print(output.predict(_input))
    """
    model.fit(train_data,
        batch_size=BATCH_SIZE,
        epochs=5,
        verbose=1,
        validation_data=test_data,
        callbacks=[checkpoint, changelr, tensorboard]
    )
    """