import fanfou
import urllib
import time
from urllib import error
import requests
from PIL import Image
from io import BytesIO
import json

psw_json = json.load(open('psw.json', encoding='utf-8'))
consumer = {'key': psw_json['download']['key'], 'secret': psw_json['download']['secret']}
client = fanfou.XAuth(consumer, psw_json['download']['username'], psw_json['download']['password'])
fanfou.bound(client)

dict_array = {}
count = 0
while 1:
    # resp = client.statuses.home_timeline()
    try:
        resp = client.statuses.public_timeline()
    except error.URLError:  # 处理返回异常
        print('return error')
    except requests.exceptions.ConnectionError:
        print('connect error')
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

    print('list-')
    for item in resp.json():  # 以此获得时间线上的20条ID
        if 'photo' in item.keys():  # 该消息中存在photo
            if 'repost_user_id' not in item.keys():  # 非转发消息
                # print(item.keys())
                # 如果url是gif图片，则跳过
                image_url = item['photo']['largeurl']
                # 如果是gif or 转发自机器人weather_image or 无聊图3号机机器人 跳过
                if image_url[-3:] == 'gif' or item['user']['id'] == 'weather_image' \
                        or item['user']['id'] == '~vB7sBb_YWyI':
                    print('this image is gif!!')
                    continue
                # 判断当前message id 是否存在于已发布的消息中。若存在，则不发布。若不存在，则 set 1
                list_id = item['id']  # type str
                if list_id in dict_array.values():
                    continue
                dict_array[count] = list_id
                count = int(count % 15) + 1
                print('count:', count)
                print(list_id, item['user']['name'], item['user']['id'], item['text'])
                print(image_url)
                # 发布转载消息
                body = {'status': '转' + item['user']['name'] + ' ' + item['text'], 'repost_status_id': item['id']}
                try:
                    # 发送消息
                    # resp = client.statuses.update(body)
                    # 保存图片到文件夹
                    response = requests.get(image_url)
                    # path = r'D:/ProjectPython/fanfou/images/' + image_url[-22:-12]
                    path = r'D:/ProjectPython/fanfou/images/' + item['user']['id'] + '+' + item['id'] + '.jpg'
                    image = Image.open(BytesIO(response.content))
                    image.save(path)
                    # urllib.urlretrieve(image_url, path)
                except error.URLError:  # 快速发布多条相同消息异常处理
                    print('400 error')
                except requests.exceptions.ConnectionError:
                    print('connect error')
                print('relay state:', resp)
    time.sleep(10)
