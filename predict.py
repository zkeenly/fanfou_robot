# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import gc
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu': 0})))
class Predict:
    def predict(image):
        gc.collect()
        norm_size = 120
        args = {}
        args['model'] = 'female.model'
        args['image'] = 'test.jpg'
        args['show'] = False
        # load the trained convolutional neural network
        print("[INFO] loading network...")
        with tf.device('/cpu:0'):
            model = load_model('female.model')
        # load the image
        orig = image.copy()
        # print(type(image))
        # pre-process the image for classification
        # 调整图像维度，变为彩色 图像
        print(image.shape)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # print(image.shape)

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

        image = cv2.resize(image, (norm_size, norm_size))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        # print('test-->2')
        # classify the input image
        # print(model)
        result = model.predict(image)[0]
        K.clear_session()
        # print('test-->3')
        # print (result.shape)
        proba = np.max(result)
        label = str(np.where(result == proba)[0])
        label_per = "{}: {:.2f}%".format(label, proba * 100)
        # print(label_per)

        if args['show']:
            print("ret")
            # draw the label on the image
            output = imutils.resize(orig, width=400)
            cv2.putText(output, label_per, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            # show the output image
            cv2.imshow("Output", output)
            cv2.waitKey(0)
        return int(label[1:2])
