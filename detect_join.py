import fanfou
import urllib
import urllib.request
from socket import *
import gc
import time
from urllib import error
import requests
import types
from PIL import Image
from io import BytesIO
from skimage import io
import numpy as np
import cv2
import json
import datetime
from dateutil.parser import parse


# 将文字编码转换为可以打印的编码
def convert_code(message_text):
    message_text = str(message_text).encode("GB18030", errors="ignore")
    message_text = message_text.decode("GBK", 'ignore')
    return message_text


# 将str 写入log文件。
def write_to_file(log_filename, log_str):
    file = open(log_filename, 'a+')
    try:
        file.write(log_str + '\n')
    except UnicodeEncodeError:
        file.write('UnicodeEncodeError:'+str(err)+'\n')
    print(log_str)


# 增加一个用户id
def add_user_id(user_id, block_filename):
    write_block_list = open(block_filename, 'a')
    write_block_list.write(user_id + '\n')
    return 1


# 搜索指定ID所在位置,从1开始
def search_user_id(user_id, block_filename):
    block_list = open(block_filename)
    position = 1
    for block_user_id in block_list:
        if user_id == block_user_id[:-1]:
            return position
        position = position + 1
    return 0


# 删除指定user_id 所在行(一次删除一个)
def del_user_id(user_id, block_filename):

    position = search_user_id(user_id, block_filename)-1
    # print(position)
    if position == -1:
        return 0
    with open(block_filename, 'r') as old_file:
        with open(block_filename, 'r+') as new_file:
            current_line = 0
            while current_line < position:
                old_file.readline()
                current_line += 1
            seek_point = old_file.tell()
            new_file.seek(seek_point, 0)
            old_file.readline()
            next_line = old_file.readline()

            while next_line:
                new_file.write(next_line)
                next_line = old_file.readline()
            new_file.truncate()
    return 1


# if return 0 except
def create_resp(client, resp_type):
    try:
        if resp_type == 'mentions':
            resp = client.statuses.mentions()
            return resp
        if resp_type == 'messages':
            resp = client.direct_messages.inbox()
            return resp
    except error.URLError:  # 处理返回异常
        print('return error join resp')
        time.sleep(10)
        return 0
    except requests.exceptions.ConnectionError:
        print('connect error')
        return 0
    except TimeoutError:
        print('time out error!')
        return 0


# process join and quit
def process_join_quit(item, block_filename, log_filename, current_date, client):
    user_id = item['user']['id']
    body = {'status': '@' + item['user']['name'] + ' The Command execute successfully, but failed to sent message.',
            'repost_status_id': item['id']}
    if item['text'] == '@深绘里 -quit':  # 处理quit
        if search_user_id(user_id, block_filename) != 0:
            write_to_file(log_filename, '-quit block_id already exist:' + user_id)
            return 0  # continue
        if add_user_id(user_id, block_filename) == 1:  # 成功添加
            write_to_file(log_filename, 'success add to block list! ' + user_id)
            content = {'user': user_id, 'text': '已经成功屏蔽检索！at:' + str(current_date)}
            try:
                client.direct_messages.new(content)
                write_to_file(log_filename, 'success post message to:' + user_id)
            except TypeError:
                # noinspection PyBroadException
                try:
                    client.statuses.update(body)
                except Exception:
                    print('message post failed!')
                write_to_file(log_filename, 'message post failed! but statuses update successful.')
            except urllib.error.HTTPError:
                # noinspection PyBroadException
                try:
                    client.statuses.update(body)
                except Exception:
                    print('message post failed!')
                write_to_file(log_filename, 'message post failed! but statuses update successful.')

    if item['text'] == '@深绘里 -join':  # 处理join
        if search_user_id(user_id, block_filename) == 0:
            write_to_file(log_filename, '-join block_id already del' + user_id)
            return 0  # continue
        if del_user_id(user_id, block_filename) == 1:  # 成功移除
            write_to_file(log_filename, 'success del from block list! ' + user_id)
            content = {'user': user_id, 'text': '已经取消屏蔽检索！at:' + str(current_date)}
            try:
                client.direct_messages.new(content)
                write_to_file(log_filename, 'success post message to:' + user_id)
            except TypeError:
                # noinspection PyBroadException
                try:
                    client.statuses.update(body)
                except Exception:
                    print('message post failed!')
                write_to_file(log_filename, 'message post failed! but statuses update successful.')
            except urllib.error.HTTPError:
                # noinspection PyBroadException
                try:
                    client.statuses.update(body)
                except Exception:
                    print('message post failed!')
                write_to_file(log_filename, 'message post failed! but statuses update successful.')


# process join and quit from msg
def process_join_quit_msg(item, block_filename, log_filename, current_date, client):
    user_id = item['sender_id']
    if item['text'] == '@深绘里 -quit':  # 处理quit
        if search_user_id(user_id, block_filename) != 0:
            write_to_file(log_filename, '-quit block_id already exist:' + user_id)
            return 0  # continue
        if add_user_id(user_id, block_filename) == 1:  # 成功添加
            write_to_file(log_filename, 'success add to block list! ' + user_id)
            content = {'user': user_id, 'text': '已经成功屏蔽检索！at:' + str(current_date)}
            try:
                client.direct_messages.new(content)
                write_to_file(log_filename, 'success post message to:' + user_id)
            except TypeError:
                write_to_file(log_filename, 'message post failed!')
            except urllib.error.HTTPError:
                write_to_file(log_filename, 'message post failed!')

    if item['text'] == '@深绘里 -join':  # 处理join
        if search_user_id(user_id, block_filename) == 0:
            write_to_file(log_filename, '-join block_id already del' + user_id)
            return 0  # continue
        if del_user_id(user_id, block_filename) == 1:  # 成功移除
            write_to_file(log_filename, 'success del from block list! ' + user_id)
            content = {'user': user_id, 'text': '已经取消屏蔽检索！at:' + str(current_date)}
            try:
                client.direct_messages.new(content)
                write_to_file(log_filename, 'success post message to :' + user_id)
            except TypeError:
                write_to_file(log_filename, 'message post failed!')
            except urllib.error.HTTPError:
                write_to_file(log_filename, 'message post failed!')


# main
class DetectJoin:

    def detect_join(log_filename):
        is_first_traver = 1  # 第一次迭代设立时间戳的标记
        is_first_traver_msg = 1
        date_flag = 0
        date_flag_msg = 0
        block_filename = 'block_list.txt'
        while 1:
            psw_json = json.load(open("psw.json", encoding='utf-8'))
            consumer = {'key': psw_json['robot']['key'], 'secret': psw_json['robot']['secret']}
            try:
                client = fanfou.XAuth(consumer, psw_json['robot']['username'], psw_json['robot']['password'])
            except TypeError:
                print('connect out of time.')
                continue
            except urllib.error.URLError:
                print('connect out of time.')
                continue
            except TimeoutError:
                print('connect out of time.')
                continue
            fanfou.bound(client)
            write_to_file(log_filename, 'start check -quit:')
            resp_mentions = create_resp(client, 'mentions')
            resp_messages = create_resp(client, 'messages')
            if resp_mentions == 0 or resp_messages == 0:
                continue
            for item in resp_mentions.json()[::-1]:  # 以此获得时间线上的20条消息
                # 设立时间flag，防止重复检索消息
                current_date = item['created_at']
                current_date = datetime.datetime.fromtimestamp(datetime.datetime.timestamp(parse(current_date)))
                if is_first_traver:  # 全局第一次执行，设立时间flag
                    is_first_traver = 0
                    date_flag = current_date
                    print('data_flag:', date_flag)
                    if (current_date - date_flag).total_seconds() < 0:  # 消息的时间早于当前的flag的时候，则跳过。
                        continue
                # print(current_date)
                elif (current_date - date_flag).total_seconds() <= 0:  # 消息的时间早于或者等于当前的flag的时候，则跳过。
                    continue
                date_flag = current_date
                print('data_flag update:', date_flag)
                print(convert_code(item['text']))
                result = process_join_quit(item, block_filename, log_filename, current_date, client)
                if result == 0:
                    continue
            for item in resp_messages.json()[::-1]:  # 以此获得时间线上的20条私信
                # 设立时间flag，防止重复检索消息
                current_date = item['created_at']
                current_date = datetime.datetime.fromtimestamp(datetime.datetime.timestamp(parse(current_date)))
                if is_first_traver_msg:  # 全局第一次执行，设立时间flag
                    is_first_traver_msg = 0
                    date_flag_msg = current_date
                    print('data_flag:', date_flag_msg)
                    if (current_date - date_flag_msg).total_seconds() < 0:  # 消息的时间早于当前的flag的时候，则跳过。
                        continue
                # print(current_date)
                elif (current_date - date_flag_msg).total_seconds() <= 0:  # 消息的时间早于或者等于当前的flag的时候，则跳过。
                    continue
                date_flag_msg = current_date
                print('data_flag update:', date_flag_msg, current_date, date_flag_msg)
                print(convert_code(item['text']))
                result = process_join_quit_msg(item, block_filename, log_filename, current_date, client)
                if result == 0:
                    continue

            time.sleep(60)
