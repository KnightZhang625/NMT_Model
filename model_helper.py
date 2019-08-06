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
from utils.log import log_error as _error

__all__ = ['get_initializer', 
           'create_emb_for_encoder_and_decoder']

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
        
def create_emb_for_encoder_and_decoder(src_vocab_size, tgt_vocab_size, embed_size, share_vocab):
    """create embedding matrinx
    """
    with tf.variable_scope('embeddings') as _:
        if share_vocab:
            if src_vocab_size != tgt_vocab_size:
                _error('src_vocab_size must be identical to tgt_vocab_size when share_vocab is True')
                raise ValueError
            _info('Using the same embedding for source and target')
            embedding = tf.get_variable('embedding', [src_vocab_size, embed_size], dtype=tf.float32)
            return embedding, embedding
        else:
            _info('Using the different embeddings for source and target respectively')
            with tf.variable_scope('encoder') as _:
                embedding_encoder = tf.get_variable(
                    'embedding_encoder', [src_vocab_size, embed_size], dtype=tf.float32)
            with tf.variable_scope('decoder') as _:
                embedding_decoder = tf.get_variable(
                    'embedding_decoder', [tgt_vocab_size, embed_size], dtype=tf.float32)
            return embedding_encoder, embedding_decoder

def create_rnn_cell(unit_type, num_units, num_layers,)


if __name__ == "__main__":
    get_initializer('dad')