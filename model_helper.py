# coding:utf-8
# Produced by Andysin Zhang
# 02_Aug_2019

# set the system path
import sys
from pathlib import Path
cur_path = Path(__file__).absolute().parent
sys.path.insert(0, str(cur_path))

import tensorflow as tf
from utils.log import log_info as _info

__all__ = ['get_initializer']

def get_initializer(init_name, seed=None, init_weight=None):
    """create the initializer
    """
    if init_name == 'uniform':
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_name == 'glorot_normal':
        return tf.keras.initializer.glorot_normal(
            seed=seed)
    elif init_name == 'glorot_uniform':
        return tf.keras.initiazer.glorot_uniform(
            seed=seed)
    else:
        _info('Unknow {} initializer'.format(init_name))
        raise ValueError
        
if __name__ == "__main__":
    get_initializer('dad')