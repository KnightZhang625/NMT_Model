# coding:utf-8
# Produced by Andysin Zhang
# 14_Aug_2019

import os
import sys
import codecs
import pickle
import random

from hparameters import hparams
from preprocess import find_num
from utils.log import log_info as _info
from utils.log import log_error as _error

# BATCH_SIZE = hparams.batch_size
BATCH_SIZE = 5

i_s = lambda s : str(s)
test = lambda l : ' '.join(list(map(i_s, l)))

def _find_files(path):
    """return the files in the given directory and the directory"""
    dir_path = path
    return dir_path, sorted(os.listdir(path), key=find_num)

def  _convert_str_to_int(string):
    """convert string to int"""
    return [int(s) for s in string.split(' ')]
    
def iter_data(path):
    """load the data with batch size for training"""
    data_bz = []
    dir_path, files_list = _find_files(path)
    
    data_temp = []
    for file_name in files_list:
        _info('Processing {}'.format(file_name))
        with codecs.open(str(dir_path / file_name), 'r', 'utf-8') as file:
            data = file.read().split('\n')[:-1]
        
        if len(data) % BATCH_SIZE != 0:
            divide_or_not = False
        else:
            divide_or_not = True

        batch_number = len(data) // BATCH_SIZE
        for bn in range(batch_number):
            data_block = data[bn * BATCH_SIZE : bn * BATCH_SIZE + BATCH_SIZE]
            data_block = list(map(_convert_str_to_int, data_block))
            yield data_block
        if not divide_or_not:
            bn +=1
            data_block = data[bn * BATCH_SIZE : ]
            try:
                data_sup = random.sample(data_block, BATCH_SIZE - len(data_block))
            except ValueError:
                data_sup = [random.choice(data_block) for _ in range(BATCH_SIZE - len(data_block))]
            data_block += data_sup
            yield data_block
    
if __name__ == '__main__':
    from pathlib import Path

    cur_path = Path(__file__).absolute().parent
    data_path = cur_path / 'data/news_data'

    for data in iter_data(data_path):
        d = list(data)
        dd = list(map(test, d))
        print(len(dd))
        print(len(set(dd)))
        input()