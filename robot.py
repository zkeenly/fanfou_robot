import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import fanfou
import urllib
import urllib.request
from socket import *
import gc
import tensorflow.python
import time
from urllib import error
import requests
import types
from PIL import Image
from io import BytesIO
from predict import Predict
from skimage import io
import io as sys_io
import numpy as np
import cv2
from detect_join import DetectJoin
import multiprocessing
import http
import json
time_sleep = 30  # 每次读取图片的暂停时间

# 将文字编码转换为可以打印的编码
def convert_code(message_text):
    message_text = str(message_text).encode("GB18030", errors="ignore")
    message_text = message_text.decode("GBK", errors="ignore")
    return message_text


# 将str 写入log文件。
def write_to_file(log_filename, log_str):
    file = open(log_filename, 'a+')
    try:
        file.write(log_str + '\n')
    except UnicodeEncodeError as err:
        file.write('UnicodeEncodeError:'+str(err)+'\n')
    print(log_str)


# 检测用户的join and quit 申请
def detect_join(traget, log_filename):
    DetectJoin.detect_join(log_filename)


# 将url图片转换为np_array格式的数组
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    # set timeout.
    try:
        resp = urllib.request.urlopen(url, data=None, timeout=5)
        # print('test--1.1')
        # bytearray将数据转换成（返回）一个新的字节数组
        # asarray 复制数据，将结构化数据转换成ndarray
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        # print('test-->1.2')
        # cv2.imdecode()函数将数据解码成Opencv图像格式
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # print('test-->1.3')
        # return the image
    except timeout:
        return 0
    else:
        return image


# 主机器人程序
def robot_start(traget, log_filename):
    print('start at:', log_filename)
    count_cycle = 30  # 总共缓存消息数量
    psw_json = json.load(open("psw.json", encoding='utf-8'))
    consumer = {'key': psw_json['robot']['key'], 'secret': psw_json['robot']['secret']}
    client = fanfou.XAuth(consumer, psw_json['robot']['username'], psw_json['robot']['password'])
    fanfou.bound(client)
    dict_array = {}  # 最近发布过的消息ID列表
    count = 0  # 最近发布过的消息ID索引
    while 1:
        # 回收内存
        gc.collect()
        # print('test-->0.1')
        # resp = client.statuses.home_timeline()
        try:
            resp = client.statuses.public_timeline()
        except error.URLError:  # 处理返回异常
            print('return error resp')
            time.sleep(10)
            continue
        except requests.exceptions.ConnectionError:
            print('connect error')
            continue
        except TimeoutError:
            print('time out error!')
            continue
        except http.client.RemoteDisconnected:
            print('RemoteDisconnected')
            continue
        # ------------------------------------
        # 打印一条消息
        '''data = resp.json()
        print(type(data))
        print(type(data[0]))
        print(data[0]['text'])
        print(data[1]['text'])'''
        # ------------------------------------
        # 发送一条饭否消息
         #body = {'status': 'hello,fanfou'}
         #resp = client.statuses.update(body)
         #print(resp.code)  # 如果成功，返回200
        # ------------------------------------

        print('-list:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        for item in resp.json():  # 获得时间线上的20条ID并逐条转发处理
            if 'photo' in item.keys():  # 该消息中存在photo
                if 'repost_user_id' not in item.keys():  # 非转发消息
                    # print('test-->0.5')
                    # print(item.keys())
                    try:

                        # ※先判断当前message id 是否存在于已检测过的消息中。若存在，则不发布。
                        list_id = item['id']  # type str
                        if list_id in dict_array.values():
                            continue

                        # 提取图片的url，如果url是gif图片，则跳过，并加入count计数器
                        image_url = item['photo']['largeurl']
                        if image_url[-3:] == 'gif':
                            dict_array[count] = item['id']
                            count = int(count % count_cycle) + 1
                            write_to_file(log_filename, 'this is gif id:count++')
                            continue

                        # 从block list中加载屏蔽用户，当存在屏蔽列表，则continue 跳过本次。
                        is_block = 0
                        block_list = open('block_list.txt')
                        # print('block_list:')
                        for block_user_id in block_list:
                            # print(block_user_id[:-1])
                            if item['user']['id'] == block_user_id[:-1]:
                                print('this id has exist in block list')
                                is_block = 1
                                break
                        if is_block == 1:
                            print('block id:', item['user']['id'])
                            dict_array[count] = item['id']
                            count = int(count % count_cycle) + 1
                            write_to_file(log_filename, 'this is block id:count++')
                            continue

                        # 提取图片 ndarray格式
                        image = url_to_image(image_url)
                        if type(image) == int:
                            image = url_to_image(image_url)
                            if type(image) == int:
                                print('get url timeout')
                                continue
                        # 如果图片过长，或者过高。对图片进行裁剪，采取上中下，左中右，以及原图四张图片预测。有一张符合则符合。
                        if len(image.shape) == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        img_hig = image.shape[0]
                        img_wide = image.shape[1]
                        # 如果图片的高度，宽度小于一定值，不预测。
                        if img_hig < 201 or img_wide < 201:
                            dict_array[count] = item['id']
                            count = int(count % count_cycle) + 1
                            write_to_file(log_filename, 'this is little image:count++')
                            continue
                        write_to_file(log_filename, '-count:' + str(count))
                        # 将信息写入log文件
                        message_text = convert_code(item['text'])
                        try:
                            write_to_file(log_filename, '\n'
                                          + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                                          + ' ' + list_id + ' ' + item['user']['name']
                                          + ' ' + item['user']['id'] + ' ' + str(message_text)
                                          + '\n' + image_url)
                        except UnicodeEncodeError as err:
                            write_to_file(log_filename, 'UnicodeEncodeError:' + str(err) + '\n')
                        # 发布转载消息
                        body = {'status': '转@' + item['user']['name'] + ' ' + item['text'], 'repost_status_id': item['id']}
                        # print('test-->1')
                        # 发送消息
                        # resp = client.statuses.update(body)
                        # 保存图片到文件夹
                        # response = requests.get(image_url)
                        # path = r'D:/ProjectPython/fanfou/images/' + image_url[-22:-12]
                        # path = r'D:/ProjectPython/fanfou/images/' + item['user']['id'] + '+' + item['id'] + '.jpg'
                        # image = Image.open(BytesIO(response.content))
                        # image.save(path)

                        # 显示图片
                        # image = io.imread(image_url)
                        # io.imshow(image)
                        # io.show()
                        # print('test-->1.5')

                        if img_wide >= img_hig * 1.4 and img_hig > 240:  # 长图片
                            img_start = int((img_wide - img_hig) / 2)
                            image_mid = image[:, img_start:img_start + img_hig, :]
                            image_left = image[:, 0:img_hig, :]
                            image_right = image[:, img_wide - img_hig:img_wide, :]
                            test_result = {}
                            test_result[0] = Predict.predict(image)
                            test_result[1] = Predict.predict(image_left)
                            test_result[2] = Predict.predict(image_mid)
                            test_result[3] = Predict.predict(image_right)
                            write_to_file(log_filename, 'long image: ' + str(test_result[0])
                                          + str(test_result[1]) + str(test_result[2]) + str(test_result[3]))
                            # 这里是四个分类的数量，0-人，1-文字，2-食物，3-动物,4-风景
                            count_classify = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                            is_human = 1
                            for i in range(4):  # 计算每个可能情况的数量（一共的数量是4）
                                key = test_result[i]
                                count_classify[key] = count_classify[key] + 1
                            for i in range(1, 5):  # 如果 1-4 中有一个出现三次可能，则不是0。
                                if count_classify[i] == 3:
                                    is_human = 0
                            if is_human and count_classify[0]:
                                resp = client.statuses.update(body)
                                # print('relay state:', resp)
                                write_to_file(log_filename, 'success post!')

                        elif img_wide * 1.4 <= img_hig and img_wide > 240:  # 高图片
                            img_start = int((img_hig - img_wide) / 2)
                            image_mid = image[img_start:img_start + img_wide, :, :]
                            image_up = image[0:img_wide, :, :]
                            image_down = image[img_hig - img_wide:img_hig, :, :]
                            test_result = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                            test_result[0] = Predict.predict(image)
                            test_result[1] = Predict.predict(image_up)
                            test_result[2] = Predict.predict(image_mid)
                            test_result[3] = Predict.predict(image_down)
                            write_to_file(log_filename, 'long image: ' + str(test_result[0])
                                          + str(test_result[1]) + str(test_result[2]) + str(test_result[3]))
                            # 这里是四个分类的数量，0-人，1-文字，2-食物，3-动物，4-风景
                            count_classify = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                            is_human = 1
                            for i in range(4):  # 计算每个可能情况的数量(一共的数量是4)
                                key = test_result[i]
                                count_classify[key] = count_classify[key] + 1
                            for i in range(1, 5):  # 如果 1-4中有一个出现三次可能，则不是0。
                                if count_classify[i] == 3:
                                    is_human = 0
                            if is_human and count_classify[0]:
                                resp = client.statuses.update(body)
                                # print('relay state:', resp)
                                write_to_file(log_filename, 'success post!')
                        else:  # 方形图片，只预测一次
                            test_result = Predict.predict(image)
                            write_to_file(log_filename, 'square image: ' + str(test_result))
                            if test_result == 0:
                                resp = client.statuses.update(body)
                                # print('relay state:', resp)
                                write_to_file(log_filename, 'success post!')

                        # urllib.urlretrieve(image_url, path)
                        dict_array[count] = list_id
                        count = int(count % count_cycle) + 1
                        write_to_file(log_filename, 'success predict:count++')
                    except error.URLError:  # 快速发布多条相同消息异常处理
                        print('400 error')
                    except requests.exceptions.ConnectionError:
                        print('connect error')
                    except ConnectionResetError:
                        print('connect reset error')
                    except MemoryError:
                        print('memory error')
                    except tensorflow.python.framework.errors_impl.ResourceExhaustedError:
                        print('resource exhausted error')
                    except tensorflow.python.framework.errors_impl.InternalError:
                        print('InternalError')
                    except ValueError:
                        print('value error')
                    except TypeError:
                        print('type error')
                    except TimeoutError:
                        print('timeout error')
                    except timeout:
                        print('socket timeout')
                    except Exception as err:
                        print(err)
                    except http.client.IncompleteRead:
                        # print('http.client.IncompleteRead error')
                        write_to_file(log_filename, 'http.client.IncompleteRead error')
                    # 没有错误，顺利执行之后，count++。不论是否发布成功
        time.sleep(time_sleep)


if __name__ == "__main__":
    # 缓存的日志文件目录/文件名
    time_suffix = time.strftime('%Y_%m_%d %H%M%S', time.localtime(time.time()))
    log_filename = 'log/log_' + str(time_suffix) + '.txt'
    sys.stdout = sys_io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码
    # 多进程处理join quit 以及机器人主程序
    p_detect = multiprocessing.Process(target=detect_join, args=(2, log_filename))
    p_robot = multiprocessing.Process(target=robot_start, args=(1, log_filename))
    p_detect.start()
    p_robot.start()

