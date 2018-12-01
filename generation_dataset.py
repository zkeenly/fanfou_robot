from imutils import paths
import cv2
from keras.preprocessing.image import img_to_array
import random
import numpy as np

path = './train'
data_name = 'train_data_240.npy'
label_name = 'train_label_240.npy'
norm_size = 240

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
    # load the image, pre-process it ,and store it in the data list
    # image = cv2.imread(imagePath)
    image = cv2.imdecode(np.fromfile(imagePath, dtype=np.float32), cv2.IMREAD_UNCHANGED)
    # 调整图像维度，变为彩色 图像
    print(imagePath)
    print(image.shape)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    print(image.shape)


    # 调整图像大小，为指定大小
    image = cv2.resize(image, (norm_size, norm_size))

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
    elif imagePath[8:15] == 'scenery':
        label = 3
        print(imagePath[8:15])
        print('textual')
    elif imagePath[8:14] == 'animal':
        label = 4
        print(imagePath[8:14])
        print('animal')
    labels.append(label)
# scale the raw pixel intensities to the range[0, 1]
# print(type(data))

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
np.save(data_name, data)
np.save(label_name, labels)
# convert the labels from integers to vectors

