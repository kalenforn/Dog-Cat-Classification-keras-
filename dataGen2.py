import os
import shutil

from keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE, BATCH_SIZE, FILE_PATH_TRAIN, FILE_PATH_TEST, DATA_PATH, DATA_SAVE_PATH

def split_data(start, end=12500):
    # 数据集解压之后的目录
    original_dataset_dir = DATA_PATH
    # 存放小数据集的目录
    base_dir = DATA_SAVE_PATH
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # 建立训练集、验证集、测试集目录
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    #validation_dir = os.path.join(base_dir, 'validation')
    #os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # 将猫狗照片按照训练、验证、测试分类
    train_cats_dir = os.path.join(train_dir, 'cats')
    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)

    train_dogs_dir = os.path.join(train_dir, 'dogs')
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)


    test_cats_dir = os.path.join(test_dir, 'cats')
    if not os.path.exists(test_cats_dir):
        os.mkdir(test_cats_dir)

    test_dogs_dir = os.path.join(test_dir, 'dogs')
    if not os.path.exists(test_dogs_dir):
        os.mkdir(test_dogs_dir)

    # 切割数据集
    fnames = ['cat.{}.jpg'.format(i) for i in range(start, end)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dat = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dat)

    fnames = ['cat.{}.jpg'.format(i) for i in range(start)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dat = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dat)

    fnames = ['dog.{}.jpg'.format(i) for i in range(start, end)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dat = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dat)

    fnames = ['dog.{}.jpg'.format(i) for i in range(start)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dat = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dat)

def genData():

    if os.path.exists(FILE_PATH_TRAIN) or os.path.exists(FILE_PATH_TEST):
        split_data(500, 12500)

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

    return (train_generator, test_generator)