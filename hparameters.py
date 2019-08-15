# coding:utf-8
# Produced by Andysin Zhang
# 08_Aug_2019

from pathlib import Path
cur_path = Path(__file__).absolute().parent
DATA_PATH = cur_path / 'data/news_data'
from utils.aux_func import NoNewAttrs

class HyperParameters(NoNewAttrs):
    """HyperParameters setting"""
    
    # Global
    model_type = 'standard'
    out_dir = './models'
    data_path = DATA_PATH
    summary_name = 'test'
    train_steps = 10000000
    batch_size = 300
    save_batch = 100
    decay_step = save_batch
    init_op = 'orthogonal'
    init_weight = 0.1

    # Learning rate
    learning_rate = 0.0005
    lr_limit = 1e-6
    decay_rate = 0.99
    random_seed = None
    warm_steps = 0

    # Update
    optimizer = 'adam'
    max_gradient_norm=5.0
    num_keep_ckpts = 3

    # Encocer & Decoder
    num_encoder_layers = 4
    num_decoder_layers = 4
    tgt_max_len_infer = None
    encoder_type = 'bi'
    unit_type = 'gru'
    num_units = int(512)
    infer_mode = 'greedy'

    attention = None

    residual = True
    dropout = 0.0
    forget_bias = 1.0
    src_vocab_size = 7819
    tgt_vocab_size = 7819
    embedding_size = 16

    # VAE
    enable_vae = True
    pre_train = True

    share_vocab = True
    beam_width = 0
    length_penalty_weight = 1.0

    sos_id = int(1)
    eos_id = int(2)
    padding_id = int(0)

hparams = HyperParameters()

