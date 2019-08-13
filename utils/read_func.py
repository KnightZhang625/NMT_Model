# coding:utf-8
# Produced by Andysin Zhang
# 13_Aug_2019

""""this file used for loading data"""

import json
import codecs
import pickle

from pathlib import Path
cur_path = Path(__file__).absolute().parent.parent
save_path = cur_path / 'data/json_seperate/'

__all__ = ['read_json']



def read_json(path):
    """read data whose format is 'json'"""
    data_json = []
    with codecs.open(path, 'r', 'utf-8') as file:
        #TODO loading the entire json will lead to 'out of memory' problem 
        for line in file:
            data_temp = json.loads(line)
            print(data_temp)
            input()            
        data = file.read().split('\n')
    # extracting the 'content' block
    data_json.extend([json.loads(d)['content'] for d in data])
    
    return data_json

def _read_json_block(file):
    """solve the problem which the Json file is too large"""
    count = 1
    threshold = 100     # divide the Json into threshold files
    for line in file:
        save_path_block = str(save_path) + 'json_block_{}'.format(str(count))
        with codecs.open(save_path_block, 'w', 'utf-8') as file:
            pass

def load_vocab(path):
    """load the vocab"""
    with codecs.open(path, 'r', 'utf-8') as file:
        vocab = file.read().split('\n')
    vocab_idx = {v : i for i, v in enumerate(vocab)}
    idx_vocab = {i : v for i, v in enumerate(vocab)}
    
    return vocab_idx, idx_vocab

if __name__ == '__main__':
    from pathlib import Path
    
    cur_path = Path(__file__).absolute().parent.parent
    data_path = cur_path / 'data/news_valid.json'
    read_json(data_path)


    '''
    data_path = cur_path / 'data/vocab.data'
    vocab_idx, idx_vocab = load_vocab(data_path)

    with codecs.open(cur_path / 'data/vocab_idx.pickle', 'wb') as file_1, \
         codecs.open(cur_path / 'data/idx_vocab.pickle', 'wb') as file_2:
         pickle.dump(vocab_idx, file_1)
         pickle.dump(idx_vocab, file_2)
    '''