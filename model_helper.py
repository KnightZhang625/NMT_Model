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

def get_initializer(init_op, random_seed=None, init_weight=None):
    """Set the initializer

    Args:
        init_op: 'orthogonal' | 'normal' | 'uniform'
    
    Returns:
        Initializer
    
    Raises:
        ValueError: Unknown initializer flag
    """
    if init_op == 'orthogonal':
        return tf.initializers.orthogonal(seed=random_seed)
    elif init_op == 'normal':
        return tf.random_normal_initializer(seed=random_seed)
    elif init_op == 'uniform':
        return tf.random_uniform_initializer(-init_weight, init_weight, seed=random_seed)
    else:
        _error('Unknown initializer {}'.format(init_op))
        raise ValueError

def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       scope):
    """Create the embedding

    Args:
        share_vocab: whether source and target use the same embedding matrix
        src_vocab_size: source vocab size
        tgt_vocab_size: target vocab size
        src_embed_size: source embedding dims
        tgt_embed_size: target embedding dims
        scope: VariableScope
    
    Returns:
        embedding_encoder & embedding_decoder
    
    Raises:
        ValueError: if share_vocab == True, src_vocab_size != tgt_vocab_size
    """
    with tf.variable_scope('embedding', dtype=tf.float32) as scope:
        if share_vocab:
            if not src_vocab_size == tgt_vocab_size:
                _error('Source vocab size must be equal to Target vocab size, however, {} to {}'
                    .format(src_vocab_size, tgt_vocab_size))
                raise ValueError
            _info('Use the same embeddings for source and target')
            embedding_encoder = tf.get_variable('embedding_shared', [src_vocab_size, src_embed_size], dtype=tf.float32)
            embedding_decoder = embedding_encoder
        else:
            with tf.variable_scope('encoder'):
                embedding_encoder = tf.get_variable('embedding_encoder', [src_vocab_size, src_embed_size], dtype=tf.float32)
            with tf.variable_scope('decoder'):
                embedding_decoder = tf.get_varialbe('embedding_decoder', [tgt_vocab_size, tgt_embed_size], dtype=tf.float32)
    return embedding_encoder, embedding_decoder

def create_rnn_cell(unit_type,
                    num_units,
                    num_layers,
                    num_residual_layers,
                    forget_bias,
                    dropout,
                    mode):
    """Create multi-layer RNN cell

    Returns:
        An RNN cell lists
    
    Raises:
        ValueError: Unknown unit type
    """
    cell_list = _create_cell_list(unit_type=unit_type,
                                  num_units=num_units,
                                  num_layers=num_layers,
                                  num_residual_layers=num_residual_layers,
                                  forget_bias=forget_bias,
                                  dropout=dropout,
                                  mode=mode)
    if len(cell_list) == 1:
        return cell_list[0]
    else:
        return tf.nn.rnn_cell.MultiRNNCell(cell_list)
    
def _create_cell_list(unit_type,
                      num_units,
                      num_layers,
                      num_residual_layers,
                      forget_bias,
                      dropout,
                      mode):
    cell_list = []
    for i in range(num_layers):
        cell = _create_single_cell(unit_type=unit_type,
                      num_units=num_units,
                      forget_bias=forget_bias,
                      dropout=dropout,
                      mode=mode,
                      residual_or_not=(i >= num_layers - num_residual_layers))
        cell_list.append(cell)
    return cell_list

def _create_single_cell(unit_type,
                        num_units,
                        forget_bias,
                        dropout,
                        mode,
                        residual_or_not):
    """Create a cell
        create a single cell, wrap with dropout and residual layer
    """
    dropout = dropout if mode == 'train' else 0.

    # 1. create a single cell first
    if unit_type == 'lstm':
        _info(' build lstm cell')
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=forget_bias)
    elif unit_type == 'gru':
        _info(' build gru cell')
        cell = tf.nn.rnn_cell.GRUCell(num_units)
    else:
        _error('Unknow unit type : {}'.format(unit_type))
        raise ValueError
    
    # Dropout
    if dropout > 0:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=(1.0-dropout))
        _info(' add dropout = {}'.format(dropout))
    
    # Residual
    if residual_or_not:
        cell = tf.nn.rnn_cell.ResidualWrapper(cell=cell)
        _info(' add residual')
    
    return cell

if __name__ == "__main__":
    pass