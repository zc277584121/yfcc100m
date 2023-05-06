# -*- coding: UTF-8 -*-
import argparse
import datetime
import os
import sys
import threading
import time
from download_repair_error import hex_range
import numpy as np
import timm


def exec_cmd(cmd):
    try:
        print("命令\n%s\n开始运行%s" % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("命令\n%s\n结束运行%s" % (cmd, datetime.datetime.now()))
    except:
        print('命令\n%s\n运行失败' % (cmd))


def exec_cmd_list(cmd_list, if_parallel=False):
    if isinstance(cmd_list, str):
        exec_cmd(cmd_list)
    elif isinstance(cmd_list, list):
        if if_parallel:
            threads = []
            for cmd in cmd_list:
                th = threading.Thread(target=exec_cmd, args=(cmd,))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()
        else:
            for cmd in cmd_list:
                exec_cmd(cmd)


interpreter = sys.executable  # todo

start = '3a0'
end = '3af'
device_list = [0, 1, 2, 3]
log_root = './logs'

if not os.path.exists(log_root):
    os.mkdir(log_root)

shard_list = hex_range(start, end)
n = len(shard_list) // len(device_list)

shard_range_list = [shard_list[i:i + n] for i in range(0, len(shard_list), n)]

if len(shard_range_list) > n:
    shard_range_list[-2] += shard_range_list[-1]
    shard_range_list.pop()

assert len(device_list) == len(shard_range_list)

cmd_list = []
for i in range(len(device_list)):
    device = device_list[i]
    start = shard_range_list[i][0]
    end = shard_range_list[i][-1]
    log_path = os.path.join(log_root, f'start_{start}-end_{end}-device_{device}.log')
    cmd = f'{interpreter} extract_embeddings_v2.py --start {start} --end {end} --device {device} > {log_path}'
    cmd_list.append(cmd)
print(cmd_list)
t0 = time.time()
exec_cmd_list(cmd_list, if_parallel=True)
t1 = time.time()
print(f'time = {t1 - t0}')
