# -*- coding: utf-8 -*-

from datetime import datetime


class Record(object):
    '''记录loss、acc等信息'''
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

class Logger(object):
    '''保存日志信息'''
    def __init__(self, lr=0, bs=0, wd=0, num_train=0):
        self.lr = lr
        self.bs = bs
        self.wd = wd
        self.num_train = num_train
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a'
        self.file = open(file, mode)
        self.file.write('\n--------------------{}--------------------\n'
                        .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.file.write('lr {}, batchsize {}, wd {}, num-train {}\n'
                        .format(self.lr, self.bs, self.wd, self.num_train))
        self.file.flush()

    def write(self, msg):
        self.file.write(msg)
        self.file.write('\n')
        self.file.flush()
    
    def close(self):
        self.file.write('---------------------------------------------\n')
        self.file.close()


