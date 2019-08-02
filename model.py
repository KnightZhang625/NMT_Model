# coding:utf-8
# Produced by Andysin Zhang
# 02_Aug_2019

'''this is the fundamental model on which our model is based on'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# set the system path
import sys
from pathlib import Path
cur_path = Path(__file__).absolute().parent
sys.path.insert(0, str(cur_path))

import abc
import tensorflow as tf

from model_helper import *

class BaseModel(object):
	"""Base Model, in order to be inheritted
	"""

	def __init__(self, hparams, mode, scope=None):
		"""All the initializations are finished here

		Args:
			hparams: Hyperparameters configurations, which are provided by hyperameters.py file
			model: TRAIN | EVAL | INFER
			scope: scope of the model
		"""
		#TODO define a function to load the parameters
		#TODO build graph, however, do not implement here
		
		self.saver = tf.train.Saver(
			tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)
	
	def _load_parameters(self, hparams, mode, scope):
		"""Load the parameters and set the initializer
		"""
		self.mode = mode
		# this is flag for distinguish between pre_train and fine tune, boolean request
		self.pre_train = hparams.pre_train
		self.dtype = tf.float32
		self.global_step = tf.Variable(0, trainable=False)

		# define the input for the model
		self.encoder_input_data = tf.placeholder(
			tf.int32, [None, None], name='encoder_input_data')
		self.decoder_input_data = tf.placeholder(
			tf.int32, [None, None], name='decoder_input_data')
		self.decoder_output_data = tf.placeholder(
			tf.int32, [None, None], name='decoder_output_data')
		self.seq_length_encoder_intput_data = tf.placeholder(
			tf.int32, [None], name='seq_length_encoder_input_data')
		self.seq_length_decoder_input_data = tf.placeholder(
			tf.int32, [None], name='seq_length_decoder_input_data')

		self.batch_size = tf.size(self.seq_length_encoder_intput_data)

		self.src_vocab_size = hparams.src_vocab_size
		self.tgt_vocab_size = hparams.tgt_vocab_size

		self.num_units = hparams.num_units
		self.num_encoder_layer = hparams.num_encoder_layer
		self.num_decoder_layer = hparams.num_decoder_layer

		# define the initializer
		initializer = get_initializer(
			hparams.init_name, hparams.random_seed, hparams.init_weight)
		tf.get_variable_scope().set_initializer(initializer)

		# TODO create embedding matrix

