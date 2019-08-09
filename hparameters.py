# coding:utf-8
# Produced by Andysin Zhang
# 08_Aug_2019

from utils.aux_func import NoNewAttrs

class HyperParameters(NoNewAttrs):
    """HyperParameters setting"""
    
    # Global
    train_steps = 10000000
    batch_size = 64
    save_batch = 3000
    decay_step = save_batch
    init_op = 'orthogonal'
    init_weight = 0.1

    # Learning rate
    learning_rate = 0.1
    lr_limit = 1e-6
    decay_rate = 0.99
    random_seed = None
    warm_steps = 0

    # Update
    optimizer = 'adam'
    max_gradient_norm=5.0
    num_keep_ckpts = 5

    # Encocer & Decoder
    num_encoder_layers = 2
    num_decoder_layers = 2
    tgt_max_len_infer = None
    encoder_type = 'bi'
    unit_type = 'gru'
    num_units = int(16)

    attention = None

    residual = True
    dropout = 0.0
    forget_bias = 1.0
    src_vocab_size = 7819
    tgt_vocab_size = 7819
    embedding_size = 16

    # VAE
    enable_vae = False
    pre_train = False

    share_vocab = True
    beam_width = 0
    length_penalty_weight = 1.0
    out_dir = './models'

    sos_id = 0
    eos_id = 1

hyper = HyperParameters()

