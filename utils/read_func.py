# coding:utf-8
# Produced by Andysin Zhang
# 13_Aug_2019

""""this file used for loading data"""

import json
import codecs
import pickle

__all__ = ['read_json']

def read_json(path):
    """read data whose format is 'json'"""
    data_json = []
    with codecs.open(path, 'r', 'utf-8') as file:
        data = file.read().split('\n')[:10]
    # extracting the 'content' block
    data_json.extend([json.loads(d)['content'] for d in data])
    
    return data_json

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
    # data_path = cur_path / 'data/news_valid.json'
    # read_json(data_path)

    data_path = cur_path / 'data/vocab.data'
    vocab_idx, idx_vocab = load_vocab(data_path)

    with codecs.open(cur_path / 'data/vocab_idx.pickle', 'wb') as file_1, \
         codecs.open(cur_path / 'data/idx_vocab.pickle', 'wb') as file_2:
         pickle.dump(vocab_idx, file_1)
         pickle.dump(idx_vocab, file_2)