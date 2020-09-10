from dataGen import genData
from keras.preprocessing.image import ImageDataGenerator
from config import FILE_PATH_TEST, FILE_PATH_TRAIN, BATCH_SIZE, IMG_SIZE
from view_result import gen_picture
def all_test():
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

    i, j = train_generator.next()
    print(i)

    train = genData("D:/data/Dog-cat/train/")
    i, j = train.__next__()
    print(i['the_input'])
    #print(train_generator)
    #print(test_generator)


if __name__ == '__main__':
    history = {"val_loss": [15.43, 10.56, 8.74, 7.62, 5.43],
               "loss": [14.25, 9.72, 7.23, 5.72, 4.32],
               "val_acc": [0.32, 0.45, 0.67, 0.75, 0.89],
               "acc": [0.21, 0.34, 0.56, 0.64, 0.87]}
    gen_picture(history, "./")