# coding:utf-8
# Produced by Andysin Zhang
# 13_Aug_2019

import os
import re
import sys
import json
import pickle
import codecs
from itertools import chain
from pathlib import Path
from utils.log import log_info as _info
from utils.log import log_error as _error

from multiprocessing import Pool

ELIMINATE_PUNCTUATION = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~（）：＊※，·… 、？！\n👍～《 》「」“”]+'
_clean = lambda line : re.sub(ELIMINATE_PUNCTUATION, '', line)

SPLIT_PUNCTUATION = '[。!！, ，；]+'
_split = lambda line : re.split(SPLIT_PUNCTUATION, line)

_remove = lambda line : line != ''

_to_int = lambda int_ : str(int_)

with codecs.open('data/vocab_idx.pickle', 'rb') as file_1, \
     codecs.open('data/idx_vocab.pickle', 'rb') as file_2:
     vocab_idx = pickle.load(file_1)
     idx_vocab = pickle.load(file_2)

cur_path = Path(__file__).absolute().parent
save_path = cur_path / 'data/news_data'

def clean_data(lines):
    """clean unnecessary punctuations"""
    lines = list(map(_clean, lines))
    return lines

    """split the sentence by necessary punctuations"""
    lines = list(map(_split, lines))
    return lines

def process(lines):
    """put the preprocess steps together"""
    lines = list(map(_clean, lines))
    lines = list(map(_split, lines))
    return lines

def combine_list(lines):
    """convert the nested lists to the clean list and remove the '' """
    results = []
    for l in lines:
        results.extend(l)
    return list(filter(_remove, results))

def start_multi(lines, func, p=6, suffix=0, write=False):
    """execute the function in multiprocesses

    Args:
        lines: data in list
        func: execute funtion
        p: process number, default is 6
        flag: as write function takes more parameters, use 'flag' to differentiate
    """
    _info('Execute by {}th processes'.format(p))
    pool = Pool(p)
    num_each_b = len(lines) // p
    
    results = []
    for i in range(p):
        if i < (p-1):
            data_b = lines[i * num_each_b : i * num_each_b + num_each_b]
        else:
            data_b = lines[i * num_each_b :]
        if not write:
            results.append(pool.apply_async(func, (data_b, )))
        else:
            pool.apply_async(func, (data_b, suffix))
    
    if not write:
        for i in range(p):
            results[i] = results[i].get()
    
    pool.close()
    pool.join()
    
    if not write:
        return list(chain(*results))

def convert_to_idx(lines):
    """convert the str to idx"""
    for idx, l in enumerate(lines):
        line_temp = []
        for v in l:
            try:
                line_temp.append(vocab_idx[v])
            except KeyError:
                line_temp.append(vocab_idx['<unk>'])
        lines[idx] = line_temp
    return lines

def wtrie_data(lines, suffix):
    """"save the data as 'save_path + data + suffix' """
    file_path = str(save_path) + '/news_{}'.format(str(suffix))
    if file_path.split('/')[-1] in os.listdir(save_path):
        _error('{} exists'.format(file_path))
        raise FileExistsError

    _info('Save {} \n'.format(file_path))
    with codecs.open(file_path, 'w', 'utf-8') as file:
        for line in lines:
            line = list(map(_to_int, line))
            file.write(' '.join(line) + '\n')
            file.flush()

def find_num(file):
    """used for extract the numbers from the string format file"""
    number = ''
    flag = False        # used for terminating the search when meeting the last number 
    for v in file:
        try:
            int(v)      # check whether the string is number or not
            number += v
            flag = True
        except ValueError:
            if flag:
                break
    return int(number)

if __name__ == '__main__':
    import os
    import time
    
    # 1. find out the Json files
    cur_path = Path(__file__).absolute().parent
    data_path = cur_path / 'data/json_seperate/'
    json_lists = sorted(os.listdir(data_path), key=find_num)    # sort the file by the actual order


    time_start = time.time()
    
    for file in json_lists:
        _info('Processsing {}'.format(file))
        suffix = find_num(file)
        with codecs.open(data_path / file, 'rb') as file:
             lines = pickle.load(file)
        # 1. clean and split the data
        lines = start_multi(lines, process)
        lines = combine_list(lines)     # IMPORTANT
        _info('Finish clean and split the data, total: {}'.format(len(lines)))
        
        # 2. convert str to idx
        lines = start_multi(lines, convert_to_idx)
        _info('Finish converting str to idx, total: {}'.format(len(lines)))
        time_end = time.time()
        _info('Processing takes {:2f}s time'.format(time_end - time_start))

        # 3. save the data
        start_multi(lines, wtrie_data, p=1, suffix=suffix, write=True)


    # # just for test
    # for i in range(2, 3):
    #     temp_path = save_path / 'news_{}'.format(i)
    #     with codecs.open(temp_path, 'r', 'utf-8') as file:
    #         lines = file.read().split('\n')[:-1]
    #         for l in lines:
    #             str = ''
    #             for idx in l.split(' '):
    #                 str += idx_vocab[int(idx)]
    #             print(str)
    #             sys.exit()
