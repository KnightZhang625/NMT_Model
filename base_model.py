# coding:utf-8
# Produced by Andysin Zhang
# 06_Aug_2019
# Inspired By Google, Appreciate for the wonderful work
#
# Copyright 2019 TCL Inc. All Rights Reserverd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""""Basic Seq2Seq model with VAE, no Attention support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import tensorflow as tf

import model_helper as _mh

from utils.log import log_info as _info
from utils.log import log_error as _error

def get_scpecific_scope_params(scope=''):
	"""used to get specific parameters for training
	"""
	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

class TrainOutputTuple(collections.namedtuple('TrainOutputTuple', 'a b c')):
	# TODO
	pass

class EvalOutputTuple(collections.namedtuple('EvalOutputTuple', 'a b')):
	# TODO
	pass

class InferOutputTuple(collections.namedtuple('InferOutputTuple', 'a')):
	# TODO
	pass

class BaseModel(object):
	"""Base Model 
	"""
	def __init__(self, hparams, mode, scope=None):
		
		self._set_params_initializer(hparams, mode, scope)
		# TODO 2. build the graph
		# TODO 3. Saver
		pass

	def _set_params_initializer(self, hparams, mode, scope):
		"""Load the parameters and set the initializer
		"""
		self.mode = mode
		# pre_train flag is used for distinguish with pre_train and fine tune
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
		
		# load some important hparamters
		self.num_units = hparams.num_units
		self.num_encoder_layers = hparams.num_encoder_layers
		self.num_decoder_layers = hparams.num_decoder_layers
		self.num_encoder_residual_layers = self.num_encoder_layers - 1
		self.num_decoder_residual_layers = self.num_decoder_layers - 1

		self.batch_size = hparams.batch_size

		# set initializer
		random_seed = hparams.random_seed
		initializer = _mh.get_initializer(hparams.init_op, random_seed, hparams.init_weight)
		tf.get_variable_scope().set_initializer(initializer)

		# embeddings
		self.src_vocab_size = hparams.src_vocab_size
		self.tgt_vocab_size = hparams.tgt_vocab_size
		self.init_embeddings(hparams, scope)

	def init_embeddings(self, hparams, scope):
		"""Init embeddings
		"""
		self.embedding_encoder, self.embedding_decoder = \
			_mh.create_emb_for_encoder_and_decoder(
				share_vocab=hparams.share_vocab,
				src_vocab_size=self.src_vocab_size,
				tgt_vocab_size=self.tgt_vocab_size,
				src_embed_size=self.num_units,
				tgt_embed_size=self.num_units,
				scope=scope)

	def build_graph(self, hparams, scope):
		_info('Start build {} graph ...'.format(self.mode))

		with tf.variable_scope('dynamic_seq2seq', dtype=self.dtype):
			# Encoder
			encoder_outputs, encoder_state = self._build_encoder(hparams)
	
	def _train_or_inference(self):
		# TODO 
		# need to optimize, etc. in train
		# used for seperate process in train and test
		pass
	
	def _build_encoder(self, hparams):
		num_layers = self.num_encoder_layers
		num_redisual_layers = self.num_encoder_residual_layers

		with tf.variable_scope('encoder') as scope:
			self.encoder_emb_inp = tf.embedding_loopup(self.embedding_encoder, self.encoder_input_data)

			if hparams.encoder_type == 'uni':
				_info('num_layers = {} num_residual_layers = {}'.format(num_layers, num_redisual_layers))
				cell = self._build_encoder_cell(hparams, num_layers, num_redisual_layers)

	def _build_encoder_cell(self, hparams, num_layers, num_residual_layers):
		"""Build a multi-layer RNN cell
		"""
		return _mh.create_rnn_cell(
			unit_type=hparams.unit_type,
			num_units=self.num_units,
			num_layers=num_layers,
			num_residual_layers=num_residual_layers
			forget_bias=hparams.forget_bias,
			dropout=hparams.dropout,
			mode=self.mode)

	def _get_infer_maximum_iterations(self):
		# TODO
		pass
	
	def _build_decoder(self):
		# TODO
		pass
	
	def _build_decoder_cell(self):
		# TODO
		pass

	def _cross_entropy_loss(self):
		# TODO
		pass

	def _vae_loss(self):
		# TODO
		pass
	
	def _compute_loss(self):
		# TODO
		pass
		
	def _get_learning_rate_warmup(self):
		# TODO
		pass
	
	def _get_learning_rate_decay(self):
		# TODO
		pass
	
	def _get_train_summary(self):
		# TODO
		pass
		
	def train(self):
		# TODO
		pass
	
	def eval(self):
		# TODO
		pass
	
	def infer(self):
		# TODO
		pass

if __name__ == '__main__':
	t = TrainOutputTuple(a='a', b='bb', c='ccc')
	print(t)