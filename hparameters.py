# coding:utf-8
# Produced by Andysin Zhang
# 08_Aug_2019

from utils.aux_func import NoNewAttrs

class HyperParameters(NoNewAttrs):
    """HyperParameters setting"""
    train_steps = 10000000
    batch_size = 64
    save_batch = 3000
    # common decat step
    decay_step = save_batch
    # adver train

    learning_rate = 1e-6
    lr_limit = 1e-6

    decay_rate = 0.99
    num_layers = 4
    decoder_extra_layers = 0
    num_units = int(1024)
    residual = True
    time_major = True
    src_vocab_size = 7819
    tgt_vocab_size = 7819
    embedding_size = 320
    encoder_type = 'bi'
    initializer_type = 'orthogonal'
    unit_type = 'gru'
    attention_mode = None
    dropout = 0.2
    forget_bias = 1.0
    share_vocab = True
    max_len_infer = None
    beam_width = 0
    length_penalty_weight = 1.0
    out_dir = './models'

hyper = HyperParameters()

