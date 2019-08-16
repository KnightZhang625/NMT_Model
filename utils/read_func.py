# coding:utf-8
# Produced by Andysin Zhang
# 13_Aug_2019

""""this file used for loading data"""

import os
import gc
import sys
import json
import codecs
import pickle

from pathlib import Path
cur_path = Path(__file__).absolute().parent.parent
save_path = cur_path / 'data/json_seperate'
sys.path.insert(0, str(cur_path))

from utils.log import log_info as _info
from utils.log import log_error as _error

__all__ = ['read_json']
 
def read_json(path):
    """read data whose format is 'json'"""
    data_json = []
    with codecs.open(path, 'r', 'utf-8') as file:
        _read_json_block(file)

    return data_json

def _read_json_block(file):
    """solve the problem where the Json file is too large"""
    count = 1
    data_temp = []
    total = 0
    save_path_block = str(save_path) + '/json_block_{}.pickle'.format(str(count))
    
    # check whether the files have been created or not
    if save_path_block.split('/')[-1] in os.listdir(save_path):
        _error('{} exits'.format(save_path_block))
        raise FileExistsError

    for line in file:
        if len(data_temp) == 25000:
            data_temp.append(json.loads(line)['content'])       # remember to add the 25001th data
            with codecs.open(save_path_block, 'wb') as file:
                pickle.dump(data_temp, file)
            _info('Save the {}'.format(save_path_block))
            count += 1
            total += len(data_temp)
            del data_temp
            gc.collect()            
            data_temp = []
            save_path_block = str(save_path) + '/json_block_{}.pickle'.format(str(count))
        else:
            try:
                data_temp.append(json.loads(line)['content'])
            except json.decoder.JSONDecodeError:
                _error('The line: {} has no <content>'.format(line))
    if len(data_temp) != 0:
        with codecs.open(save_path_block, 'wb') as file:
            pickle.dump(data_temp, file)
        _info('Save the {}'.format(save_path_block))        
    total += len(data_temp)   

def load_vocab(path):
    """load the vocab"""
    with codecs.open(path, 'r', 'utf-8') as file:
        vocab = file.read().split('\n')
    vocab_idx = {v : i for i, v in enumerate(vocab)}
    idx_vocab = {i : v for i, v in enumerate(vocab)}
    
    return vocab_idx, idx_vocab

if __name__ == '__main__':
    from pathlib import Path
    
    # load the Json and seperate it into different files
    cur_path = Path(__file__).absolute().parent.parent
    # data_path = cur_path / 'data/news_train.json'
    # read_json(data_path)

    data_path = cur_path / 'data/vocab.data'
    vocab_idx, idx_vocab = load_vocab(data_path)

    with codecs.open(cur_path / 'data/vocab_idx.pkl', 'wb') as file_1, \
         codecs.open(cur_path / 'data/idx_vocab.pkl', 'wb') as file_2:
         pickle.dump(vocab_idx, file_1, protocol=2)
         pickle.dump(idx_vocab, file_2, protocol=2)

    # json_path = cur_path / 'data/json_seperate/json_block_4.pickle'
    # with codecs.open(json_path, 'rb') as file:
    #     data = pickle.load(file)
    #     print(data[:10])