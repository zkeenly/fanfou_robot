import matplotlib
import sys
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from lenet import LeNet
from res50 import Res50
from vgg16 import VGG16
import tensorflow as tf

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test", required=True,
                    help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset-train", required=True,
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args

# initialize the number of epochs to train for, initial learning rate
# and batch size


EPOCHS = 100
INIT_LR = 2e-4
BS = 48
CLASS_NUM = 5
norm_size = 120


def load_data_from_npy(train_type):
    print('from file generation ', train_type)
    if norm_size == 240:
        if train_type == 'train':
            data = np.load('train_data_240.npy')
            labels = np.load('train_label_240.npy')
            labels = to_categorical(labels,num_classes=CLASS_NUM)
        elif train_type == 'test':
            data = np.load('test_data_240.npy')
            labels = np.load('test_label_240.npy')
            labels = to_categorical(labels,num_classes=CLASS_NUM)
    elif norm_size == 360:
        if train_type == 'train':
            data = np.load('train_data.npy')
            labels = np.load('train_label.npy')
            labels = to_categorical(labels,num_classes=CLASS_NUM)
        elif train_type == 'test':
            data = np.load('test_data.npy')
            labels = np.load('test_label.npy')
            labels = to_categorical(labels,num_classes=CLASS_NUM)
    return data, labels
def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    print(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        #load the image, pre-process it ,and store it in the data list
        # image = cv2.imread(imagePath)
        image = cv2.imdecode(np.fromfile(imagePath, dtype=np.float32), cv2.IMREAD_UNCHANGED)
        # 调整图像维度，变为彩色 图像
        print(imagePath)
        print(image.shape)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        print(image.shape)
        # 调整图像大小，调节为中心区域正方图像
        '''print('see:', image.shape)
        img_hig = image.shape[0]
        img_wide = image.shape[1]
        if img_wide >= img_hig:  # 长图片
            img_start = int((img_wide - img_hig) / 2)
            image = image[:, img_start:img_start + img_hig, :]
            print('see2:', image.shape)
        if img_wide < img_hig:  # 高图片
            img_start = int((img_hig - img_wide) / 2)
            image = image[img_start:img_start + img_wide, :, :]
            print('see3:', image.shape)'''

        # 调整图像大小，为指定大小
        image = cv2.resize(image, (norm_size, norm_size))

        '''cv2.namedWindow("Image")
        cv2.imshow("test", image)
        cv2.waitKey(0)'''
        # 将图像数据存储到data中
        image = img_to_array(image)
        print(type(image))
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        # label = int(imagePath.split(os.path.sep)[-2])
        if imagePath[8:14] == 'female':
            label = 0
            print(imagePath[8:14])
            print('people')
        elif imagePath[8:15] == 'textual':
            label = 1
            print(imagePath[8:15])
            print('textual')
        elif imagePath[8:13] == 'foods':
            label = 2
            print(imagePath[8:13])
            print('foods')
        elif imagePath[8:14] == 'animal':
            label = 3
            print(imagePath[8:14])
            print('animal')
        elif imagePath[8:15] == 'scenery':
            label = 4
            print(imagePath[8:15])
            print('scenery')
        labels.append(label)
    # scale the raw pixel intensities to the range[0, 1]
    # print(type(data))

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    # np.save('data.npy', data)
    # np.save('label.npy', labels)
    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels

def train(aug, trainX, trainY, testX,textY,args):
    # initialize the model
    print("[INFO] compiling model...")
    # 选择模型。/Lenet OR Res50
    # model = LeNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    # model = Res50.ResNet50(input_shape=(norm_size, norm_size, 3), classes=CLASS_NUM)
    model = VGG16.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    # train the network
    print("[INFO] training network..")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


# python train.py --dataset_train ../../traffic-sign/train --dataset_test ../../traffic-sign/test --model traffic_sign.model
if __name__ == '__main__':
    '''args = args_parse()
    train_file_path = args["dataset_train"]
    test_file_path = args["dataset_test"]'''
    args = {}
    args["model"] = 'female.model'
    args["plot"] = 'loss_image'
    # from dataset generation data
    trainX, trainY = load_data('./train')
    testX, testY = load_data('./test_')
    # from file generation data
    #trainX, trainY = load_data_from_npy('train')
    #testX, testY = load_data_from_npy('test')
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)
    train(aug, trainX, trainY, testX, testY, args)

