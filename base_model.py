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
import functools
import collections
import tensorflow as tf

import model_helper as _mh

from utils.log import log_info as _info
from utils.log import log_error as _error

def get_scpecific_scope_params(scope=''):
	"""used to get specific parameters for training
	"""
	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

class TrainOutputTuple(collections.namedtuple('TrainOutputTuple', 
		'train_loss predict_count global_step batch_size learning_rate')):
	pass

class EvalOutputTuple(collections.namedtuple('EvalOutputTuple', 
		'eval_loss predict_count batch_size')):
	pass

class InferOutputTuple(collections.namedtuple('InferOutputTuple', 
		'infer_logits sample_id')):
	pass

class BaseModel(object):
	"""Base Model 
	"""
	def __init__(self, hparams, mode, scope=None):
		
		# load parameters
		self._set_params_initializer(hparams, mode, scope)
		# build graph
		res = self.build_graph(hparams, scope)
		# optimizer or infer
		self._train_or_inference(hparams, res)
	
		# Saver
		self.saver = tf.train.Saver(
			tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

	def _set_params_initializer(self, hparams, mode, scope):
		"""Load the parameters and set the initializer
		"""
		self.mode = mode
		# pre_train flag is used for distinguish with pre_train and fine tune
		if hparams.enable_vae:
			_info('Enable VAE')
			self.enable_vae = True
			self.pre_train = hparams.pre_train
		else:
			self.enable_vae = False
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

		with tf.variabel_scope('decoder/output'):
			self.output_layer = tf.layers.Dense(
				self.tgt_vocab_size, use_bias=False, name='output_layer')

		with tf.variable_scope('dynamic_seq2seq', dtype=self.dtype):
			# Encoder
			encoder_outputs, encoder_state = self._build_encoder(hparams)

			# vae check
			# when fine tune, the model need to be restored from the pre_train model,
			# if using naive 'if else', the variable in fine tune will not be built actually,
			# so that the restore will not be successful when change the pre_train to False,
			# it is necessarry to use tf.cond
			if sefl.enable_vae:
				encoder_state = tf.cond(self.pre_train, 
										lambda : self._vae_pre_train(encoder_state), 
										lambda : self._vae_fine_tune(encoder_state))
			
			# Decoder
			logits, sample_id, _ = self._build_decoder(encoder_outputs, encoder_state, hparams)
			
			# Loss
			if self.mode != 'infer':
				loss, loss_per_token, kl_loss = self._compute_loss(logits)
			else:
				loss = 0.
		
		return logits, sample_id, loss

	def _train_or_inference(self, hparams, res):
		"""need to optimize, etc. in train,
		   used for seperate process in train and test
		"""
		if self.mode == 'train':
			self.loss = res[2]
		elif self.mode == 'eval':
			self.loss = res[2]
		elif self.mode == 'infer':
			self.infer_logtis, self.sample_id = res[0], res[1]

		if self.mode != 'infer':
			self.predict_count = tf.reduce_sum(self.seq_length_encoder_intput_data)
		
		if self.enable_vae and self.pre_train:
			params = get_scpecific_scope_params('dynamic_seq2seq/transfer')
		else:
			params = tf.trainable_variables()
		
		# set learning rate
		if self.mode == 'train':
			self.learning_rate = tf.constant(hparams.learning_rate)
			# warm-up or decay
			self.learning_rate = self._get_learning_rate_warmup_decay(hparams)

			# Optimier
			if hparams.optimizer == 'sgd':
				opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			elif hparams.optimizer == 'adam':
				opt = tf.train.AdamOptimizer(self.learning_rate)
			else:
				_error('Unknown optimizer type {}'.format(hparams.optimizer))
				raise ValueError
			
			# Gradients
			gradients = tf.gradients(self.loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(
				gradients, max_gradient_norm=5.0)
			self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
		
	def _build_encoder(self, hparams):
		"""Build the encoder and return the encoding outputs

		Args:
			hparams: hyperameters
		
		Returns:
			encoder_outputs: 'uni': [batch, seq, hidden] 'bi': [batch, seq, hidden * 2]
			encoder_state: 'uni': [batch, hidden] for _ in range(layers)
						   'bi': Tuple(fw_hidden_layer_i, bw_hidden_layer_i for i in range(layers))

		Raises:
			ValueError: Unknown encoder_type
		"""
		num_layers = self.num_encoder_layers
		num_redisual_layers = self.num_encoder_residual_layers

		with tf.variable_scope('encoder') as scope:
			self.encoder_emb_inp = tf.embedding_loopup(self.embedding_encoder, self.encoder_input_data)

			if hparams.encoder_type == 'uni':
				_info('num_layers = {} num_residual_layers = {}'.format(num_layers, num_redisual_layers))
				# 1. build a list of cells
				cell = self._build_encoder_cell(hparams, num_layers, num_redisual_layers)
				# 2. forward
				# encoder_outputs: [batch, time, hidden]
				# encoder_state: ([batch, hidden] for _ in range(layers))
				encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
					cell,
					self.encoder_emb_inp,
					dtype=self.dtype,
					sequence_length=self.seq_length_encoder_intput_data,
					swap_memory=True)
			elif hparams.encoder_type == 'bi':
				if not num_layers % 2 == 0:
					_error('Bi-directional requires num_layers={} should be divided by 2'.format(num_layers))
					raise ValueError
				num_bi_layers = int(num_layers / 2)
				num_bi_residual_layers = num_bi_layers - 1
				_info(' num_bi_layers={} num_bi_residual_layers={}'.format(num_bi_layers, num_bi_residual_layers))

				cell_fw = self._build_encoder_cell(hparams, num_bi_layers, num_bi_residual_layers)
				cell_bw = self._build_encoder_cell(hparams, num_bi_layers, num_bi_residual_layers)

				# bi_outputs: (fw, bw): fw: [batch, seq, hidden]
				# bi_state: (fw, bw): fw : [[batch, hidden] for _ in range(layers)]
				bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
					cell_fw,
					cell_bw,
					dtype=self.dtype,
					sequence_length=self.seq_length_encoder_intput_data,
					swap_memory=True)

				if num_bi_layers == 1:
					encoder_state = bi_state
				else:
					encoder_state = []
					for layer_id in range(num_bi_layers):
						encoder_state.append(bi_state[0][layer_id])		# fw state in layer id
						encoder_state.append(bi_state[1][layer_id])		# bw state in layer id
					encoder_state = tuple(encoder_state)
				encoder_outputs = tf.concat(bi_outputs, -1)		# [batch, seq, hidden * 2]
			else:
				_error('Unknow encoder type: {}'.format(hparams.encoder_type))
				raise ValueError
		
		return encoder_outputs, encoder_state
				
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

	def _combine_encoder_state(func):
		@functools.wraps(func)
		def _combine_encoder_state_inner(self, encoder_state):
			encoder_state = list(encoder_state)
			encoder_state = tf.transpose(encoder_state, [1, 0, 2])	# [batch, num_layers, hidden]
			sample_norm_shape = tf.shape(encoder_state)
			z = func(self, encoder_state, sample_norm_shape)
			_info('finish sample z: {}'.format(z))
			return tuple(tf.tranpose(z, [1, 0, 2]))		# [num_layers, batch, hidden]
		return _combine_encoder_state_inner

	@_combine_encoder_state
	def _vae_pre_train(*args, **kwargs):
		_info('building pre_train step for vae')
		self = args[0]
		encoder_state= args[1]
		sample_norm_shape = args[2]
		with tf.variabel_scope('pre_train'):
			self.mean_pre_train_matrix = tf.layers.Dense(self.num_units, activation=tf.nn.relu)
			self.logVar_pre_train_matrix = tf.layers.Dense(self.num_units, activation=tf.nn.relu)
			self.mean_pre_train = self.mean_pre_train_matrix(encoder_state)
			self.logVar_pre_train = self.logVar_pre_train_matrix(encoder_state)
		eps = tf.random_normal([sample_norm_shape[0], sample_norm_shape[1], self.num_units], 0.0, 1.0, dtypt=tf.float32)		
		z = self.mean_pre_train + tf.exp(0.5 * self.logVar_pre_train) * eps
		return z

	@_combine_encoder_state
	def _vae_fine_tune(*args, **kwargs):
		_info('building fune_tune step for vae')
		self = args[0]
		encoder_state = args[1]
		sample_norm_shape = args[2]
		with tf.variable_scope('pre_train', reuse=True):
			mean_fine_tune = self.mean_pre_train_matrix(encoder_state)
			logVar_fine_tune = self.logVar_pre_train_matrix(encoder_state)
		with tf.variabel_scope('transfer'):
			self.transfer_mean = tf.layers.Dense(self.num_units, activation=tf.nn.relu)
			self.transfer_logVar = tf.layers.Dense(self.num_units, activation=tf.nn.relu)
			self.mean_transfer = self.transfer_mean(mean_fine_tune)
			self.logVar_trainsfer = self.transfer_logVar(logVar_fine_tune)
		eps = tf.random_normal([sample_norm_shape[0], sample_norm_shape[1], self.num_units], 0.0, 1.0, dtypt=tf.float32)		
		z = self.mean_transfer + tf.exp(0.5 * self.logVar_trainsfer) * eps
		return z

	def _get_infer_maximum_iterations(self, hparams):
		"""Maximum decoding steps at inference time
		"""
		if hparams.tgt_max_len_infer:
			maximum_iterations = hparams.tgt_max_len_infer
			_info('decoding with maximum iterations {}'.format(maximum_iterations))
		else:
			decoding_length_factor = 3.0
			max_encoder_length = tf.reduce_max(self.seq_length_decoder_input_data)
			maximum_iterations = tf.to_int32(tf.round(
				tf.to_float(max_encoder_length) * decoder_length_factor))
		return maximum_iterations
	
	def _build_decoder(self, encoder_outputs, encoder_state, hparams):
		"""Build decoder and return results

		Args:
			encoder_outputs: the outputs from the encoder, [batch, time, hidden] or [batch, time, hidden * 2]
			encoder_state: the final state of the encoder, [b, h]([b, h_f], [b, h_b]) for _ in range(layers)

		Returns:
			logits: [batch, time, vocab_size]
		
		Raises:
			ValueError: Unknown infer mode
		"""
		tgt_sos_id = tf.cast(tf.constant(hparams.sos_id), tf.int32)
    	tgt_eos_id = tf.cast(tf.constant(hparams.eos_id), tf.int32)

		maximum_iterations = self._get_infer_maximum_iterations(hparams)

		# Decoder
		with tf.variable_scope('decoder') as decoder_scope:
			cell, decoder_initial_state = self._build_decoder_cell(
				hparams, encoder_outputs, encoder_state)
			
			logits = tf.np_op()
			decoder_outputs = None

			# Train or Eval
			if self.mode != 'infer':
				decoder_emb_input = tf.nn.embedding_lookup(self.embedding_decoder, self.decoder_input_data)

				# helper
				helper = tf.contrib.seq2seq.TrainingHelper(
					decoder_emb_input, self.seq_length_decoder_input_data)
				
				# decoder
				my_decoder = tf.contrib.seq2seq.BasicDecoder(
					cell,
					helper,
					decoder_initial_state)
				
				# dynamic decoding
				outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
					my_decoder,
					swap_memory=True,
					scope=decoder_scope)
				
				sample_id = outputs.sample_id
				logits = self.output_layer(outputs.rnn_output)
			else:
				infer_mode = hparams.infer_mode
				start_tokens = tf.fill([self.batch_size], tgt_sos_id)
				end_token = tgt_eos_id
				_info(' decoder by infer_mode={} beam_width={}'.format(infer_mode, hparams.beam_width))

				if infer_mode == 'greedy':
					helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
						self.embedding_decoder, start_tokens, end_token)
				elif infer_mode == 'beam_search':
					beam_width = hparams.beam_width
					length_penalty_weight = hparams.length_penalty_weight
					coverage_penalty_weight = hparams.coverage_penalty_weight

					# beam search do not require helper
					my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
						cell=cell,
						embedding=self.embedding_decoder,
						start_tokens=start_tokens,
						end_token=end_token,
						initial_state=decoder_initial_state,
						beam_width=beam_width,
						output_layer=self.output_layer,
						length_penalty_weight=length_penalty_weight,
						coverage_penalty_weight=coverage_penalty_weight)
				else:
					_error('Unknown infer_mode {}'.format(infer_mode))
					raise ValueError
				
				if infer_mode != 'beam_search':
					my_decoder = tf.contrib.seq2seq.BasicDecoder(
						cell,
						helper,
						decoder_initial_state,
						output_layer=self.output_layer)		# apply to the RNN output prior to storing the result or sampling
				
				outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
					my_decoder,
					maximum_iterations=maximum_iterations,
					swap_memory=True,
					scope=decoder_scope)
			
				if infer_mode == 'beam_search':
					sample_id = outputs.predicted_ids
				else:
					logits = outputs.rnn_output
					sample_id = outputs.sample_id

		return logits, sample_id, final_context_state

	def _build_decoder_cell(self, encoder_state):
		"""build RNN cell
		"""
		if hparams.attention:
			_error('The basic model does not support Attention')
			raise ValueError
		cell = _mh.create_rnn_cell(
			unit_type=self.unit_type,
			num_units=self.num_units,
			num_layers=self.num_decoder_layers,
			num_residual_layers=self.num_decoder_residual_layers,
			forget_bias=hparams.forget_bias,
			dropout=hparams.dropout,
			model=self.mode)
		
		if self.mode == 'infer' and hparams.infer_mode == 'beam_search':
			decoder_initial_state = tf.contrib.seq2seq.tile_batch(
				encoder_state, multiplier=hparams.beam_width)
		else:
			decoder_initial_state = encoder_state
		
		return cell, decoder_initial_state

	def _vae_loss(self):
		return -0.5 * tf.reduce_sum(1.0 + self.logVar_pre_train - tf.square(self.mean_pre_train) - tf.exp(self.logVar_pre_train)) / self.batch_size) * 0.001	
	
	def _compute_loss(self, logtis):
		"""Compute loss"""
		# compute cross-entropy loss
		max_time = tf.shape(self.decoder_output_data)[1]
		ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.decoder_output_data, logtis=logtis)
		target_weights = tf.sequence_mask(
			self.seq_length_decoder_input_data,
			max_time,
			dtype=tf.float32)
		ce_loss_clear = tf.reduce_sum(ce_loss * target_weights) / self.batch_size

		# vae loss
		kl_loss = tf.cond(self.pre_train, lambda : self._vae_loss, lambda: 0.)

		# total loss
		loss_per_token = ce_loss_clear / tf.reduce_sum(target_weights)
		loss = ce_loss_clear + kl_loss
		
		return loss, loss_per_token, kl_loss

	def _get_learning_rate_warmup_decay(self, hparams):
		"""warmup or decay learning rate"""
		warm_steps = hparams.warm_steps
		warmup_factor = tf.exp(tf.log(0.01) / warm_steps)
		inv_decay = warmup_factor ** (tf.to_float(warm_steps - self.global_step))
		return tf.cond(self.global_step < hparams.warm_steps,
					   lambda : inv_decay * self.learning_rate,
					   lambda : tf.train.exponential_decay(self.learning_rate,
					   									   self.global_step,
														   hparams.decay_step,
														   hparams.decay_rate,
														   staircase=True),
					   name='learning_rate_warm_decay_cond')
	
	def _get_train_summary(self):
		# TODO
		pass
		
	def train(self, sess, feed_dict):
		"""Build train graph"""
		assert self.mode == 'train'
		output_tuple = TrainOutputTuple(train_loss=self.loss,
										predict_count=self.predict_count,
										global_step=self.global_step,
										batch_size=self.batch_size,
										self.learning_rate=learning_rate)
		return sess.run([self.update, output_tuple], feed_dict=feed_dict)
	
	def eval(self, sess, feed_dict):
		"""Build eval graph"""
		assert self.mode == 'eval'
		output_tuple = EvalOutputTuple(eval_loss=self.loss,
									   predict_count=self.predict_count,
									   batch_size=self.batch_size)
		return sess.run([self.update, output_tuple], feed_dict=feed_dict)

	def infer(self, sess, feed_dict):
		assert self.mode == 'infer'
		output_tuple = InferOutputTuple(infer_logits=self.infer_logtis,
										sample_id=self.sample_id)
		return sess.run(output_tuple,feed_dict=feed_dict)

if __name__ == '__main__':
	pass