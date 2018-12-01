from imutils import paths
import cv2
import numpy as np
import random
import scipy.misc

path = './pre_train'

imagePaths = sorted(list(paths.list_images(path)))
random.seed(42)
random.shuffle(imagePaths)
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it ,and store it in the data list
    # image = cv2.imread(imagePath)
    image = cv2.imdecode(np.fromfile(imagePath, dtype=np.float32), cv2.IMREAD_UNCHANGED)
    # 调整图像维度，变为e彩色 图像
    print(imagePath)
    print(image.shape)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 对图像进行裁剪。
    img_hig = image.shape[0]
    img_wide = image.shape[1]
    if img_wide >= img_hig * 1.5:  # 长图片
        img_start = int((img_wide - img_hig) / 2)
        image_mid = image[:, img_start:img_start + img_hig, :]
        image_left = image[:, 0:img_hig, :]
        image_right = image[:, img_wide - img_hig:img_wide, :]
        print('see2:', image.shape)
        print(image_mid.shape)
        print(image_left.shape)
        print(image_right.shape)
        cv2.imwrite('./pre_train/process/' + imagePath[11:-4] + '_mid.jpg', image_mid)
        cv2.imwrite('./pre_train/process/' + imagePath[11:-4] + '_left.jpg', image_left)
        cv2.imwrite('./pre_train/process/' + imagePath[11:-4] + '_right.jpg', image_right)
    elif img_wide * 1.5 < img_hig:  # 高图片
        img_start = int((img_hig - img_wide) / 2)
        image_mid = image[img_start:img_start + img_wide, :, :]
        image_up = image[0:img_wide, :, :]
        image_down = image[img_hig - img_wide:img_hig, :, :]
        print('see3:', image.shape)
        print(image_mid.shape)
        print(image_up.shape)
        print(image_down.shape)
        cv2.imwrite('./pre_train/process/' + imagePath[11:-4] + '_mid.jpg', image_mid)
        cv2.imwrite('./pre_train/process/' + imagePath[11:-4] + '_up.jpg', image_up)
        cv2.imwrite('./pre_train/process/' + imagePath[11:-4] + '_down.jpg', image_down)
    else:
        cv2.imwrite('./pre_train/process/' + imagePath[11:], image)
