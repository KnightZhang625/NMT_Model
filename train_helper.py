# coding:utf-8
# Produced by Andysin Zhang
# 14_Aug_2019

import os
import sys
import codecs
import pickle
import random
import collections
import tensorflow as tf
import numpy as np
from pathlib import Path

from hparameters import hparams
from preprocess import find_num
from utils.log import log_info as _info
from utils.log import log_error as _error

cur_path = Path(__file__).absolute().parent

__file__ = ['iter_data']

BATCH_SIZE = hparams.batch_size
SOS = hparams.sos_id
EOS = hparams.eos_id
PAD = hparams.padding_id
pre_train = hparams.pre_train

class DataTuple(collections.namedtuple('data',
                                'encoder_input_data \
                                 decoder_input_data \
                                 decoder_output_data \
                                 seq_length_encoder_data \
                                 seq_length_decoder_data')):
    pass

class QueAnsTuple(collections.namedtuple('que_ans',
                                    'question \
                                     answer')):
    pass

def _find_files(path):
    """return the files in the given directory and the directory"""
    dir_path = path
    return dir_path, sorted(os.listdir(path), key=find_num)

def _convert_str_to_int_fine_tune(data_tuple):
    """different solution to the fine tune step"""
    question_int = _convert_str_to_int(data_tuple.question, True)
    answer_int = _convert_str_to_int(data_tuple.answer, True)
    return QueAnsTuple(question=question_int, answer=answer_int)

def  _convert_str_to_int(string, pre_train=pre_train):
    """convert string to int"""
    if pre_train:
        return [int(s) for s in string.split(' ')]
    else:
        return _convert_str_to_int_fine_tune(string)

insert_sos = lambda l : [SOS] + l
insert_eos = lambda l : l + [EOS]

def _create_data(data, pre_train=pre_train):
    """used for creating inputs, outputs, seq_len"""
    if pre_train:
        encoder_input_data = data
        decoder_input_data = list(map(insert_sos, data))
        decoder_output_data = list(map(insert_eos, data))
    else:
        encoder_input_data = [d.question for d in data]
        decoder_input_data = list(map(insert_sos, [d.answer for d in data]))
        decoder_output_data = list(map(insert_eos, [d.answer for d in data]))
    seq_length_encoder_data = [len(d) for d in encoder_input_data]
    seq_length_decoder_data = [len(d) for d in decoder_input_data]

    encoder_input_data = np.array(tf.keras.preprocessing.sequence.pad_sequences(
        encoder_input_data, padding='post', value=PAD))
    decoder_input_data = np.array(tf.keras.preprocessing.sequence.pad_sequences(
        decoder_input_data, padding='post', value=PAD))
    decoder_output_data = np.array(tf.keras.preprocessing.sequence.pad_sequences(
        decoder_output_data, padding='post', value=PAD))

    data_tuple = DataTuple(encoder_input_data=encoder_input_data,
                           decoder_input_data=decoder_input_data,
                           decoder_output_data=decoder_output_data,
                           seq_length_encoder_data=seq_length_encoder_data,
                           seq_length_decoder_data=seq_length_decoder_data)
    return data_tuple

_split_que_ans = lambda line : QueAnsTuple(question=line.split('=')[0],
                                           answer=line.split('=')[1])

def iter_data(path, pre_train=pre_train):
    """load the data with batch size for training"""
    if pre_train:
        data_bz = []
        dir_path, files_list = _find_files(path)
        
        data_temp = []
        for file_name in files_list:
            _info('Processing {}'.format(file_name))
            with codecs.open(str(dir_path / file_name), 'r', 'utf-8') as file:
                data = file.read().split('\n')[:-1]
            
            # check whether the length of data could be divided by the batch size or not
            if len(data) % BATCH_SIZE != 0:
                divide_or_not = False
            else:
                divide_or_not = True

            batch_number = len(data) // BATCH_SIZE
            for bn in range(batch_number):
                data_block = data[bn * BATCH_SIZE : bn * BATCH_SIZE + BATCH_SIZE]
                data_block = list(map(_convert_str_to_int, data_block))
                yield _create_data(data_block)
            # if not, random sample data from the rest to let the length equal to the batch size
            if not divide_or_not:
                bn = batch_number
                data_block = data[bn * BATCH_SIZE : ]
                try:
                    data_sup = random.sample(data_block, BATCH_SIZE - len(data_block))
                except ValueError:
                    # except the circumstance where the sample number is larger then the original length
                    data_sup = [random.choice(data_block) for _ in range(BATCH_SIZE - len(data_block))]
                data_block += data_sup
                data_block = list(map(_convert_str_to_int, data_block))
                yield _create_data(data_block)
    else:
        # open the file, split the files to question and answer
        with codecs.open(path, 'r', 'utf-8') as file:
            lines = file.read().split('\n')
            try:
                lines = list(map(_split_que_ans, lines))
            except IndexError:
                _error('please delete the last line from the {}'.format(path), head='Index Error')
                raise IndexError
            _info('Successful open {}'.format(path), 'OPEN FILE')
        # create batches and return them      
        batch_number = len(lines) // BATCH_SIZE
        if len(lines) % BATCH_SIZE == 0:
            divide_or_not = True
        else:
            divide_or_not = False
        
        for bn in range(batch_number):
            lines_block = lines[bn * BATCH_SIZE : bn * BATCH_SIZE + BATCH_SIZE]
            for idx, line in enumerate(lines_block):
                lines_block[idx] = _convert_str_to_int(line, pre_train)
            yield _create_data(lines_block, pre_train)

        if not divide_or_not:
            bn = batch_number
            lines_block = lines[bn * BATCH_SIZE : ]
            try:
                lines_sup = random.sample(lines_block, BATCH_SIZE - len(lines_block))
            except ValueError:
                # except the circumstance where the sample number is larger then the original length
                lines_sup = [random.choice(lines_block) for _ in range(BATCH_SIZE - len(lines_block))]
            lines_block += lines_sup
            for idx, line in enumerate(lines_block):
                lines_block[idx] = _convert_str_to_int(line, pre_train)
            yield _create_data(lines_block, pre_train)

if __name__ == '__main__':
    # data_path = cur_path / 'data/news_data'

    # i_s = lambda s : str(s)
    # test = lambda l : ' '.join(list(map(i_s, l)))

    # for data in iter_data(data_path):
    #     print('success')

    for data in iter_data('data/chat_idx.txt', pre_train=False):
        print(data)
        input()