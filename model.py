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
import collections

from model_helper import *

from utils.log import log_info as _info
from utils.log import log_error as _error

def get_specific_scope_params(scope=''):
	return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

class TrainOutputTuple(collections.namedtuple(
	'TrainOutputTuple', ('train_summary', 'train_loss', 'learning_rate'))):
	pass

class EvalOutputTuple(collections.namedtuple(
	'EvalOutputTuple', ('eval_loss'))):
	pass

class InferOutputTuple(collections.namedtuple(
	'InferOutputTuple', ('sample_id'))):
	pass

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
		self._load_parameters(hparams, mode, scope)
		#logits, loss, final_context_state, sample_id
		res = self._build_graph(self, scope)
		if self.mode == 'TRAIN':
			self.train_loss = res[1]
		elif self.mode == 'EVAL':
			self.eval_loss = res[1]
		elif self.mode == 'INFER':
			self.sample_id = res[3]
		
		params = tf.trainable_variables()
		#TODO update parameters according to the specified scope

		if self.mode == 'TRAIN':
			self.learning_rate = tf.constant(hparams.learning_rate)
			# warm-up
			self.learning_rate = self._get_lr_warmup(hparams)
			self.learning_rate = self._get_lr_decay(hparams)

			if hparams.optimizer == 'adam':
				opt = tf.train.AdamOptimizer(self.learning_rate)
			else:
				_error('Unknown optimizer {}'.format(hparams.optimizer))
				raise ValueError
			#TODO implement other optimizer methods

			# Gradients
			gradients = tf.gradients(self.train_loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(
				gradients, max_gradient_norm=5.0)
			self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
			self.train_summary = self._get_train_summary()
		elif self.mode == 'INFER':
			self.infer_summary = self._get_infer_summary(hparams)

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

		self.num_units = hparams.num_units
		self.num_encoder_layer = hparams.num_encoder_layer
		self.num_decoder_layer = hparams.num_decoder_layer

		# define the initializer
		initializer = get_initializer(
			hparams.init_name, hparams.random_seed, hparams.init_weight)
		tf.get_variable_scope().set_initializer(initializer)

		# create embedding layer
		self.src_vocab_size = hparams.src_vocab_size
		self.tgt_vocab_size = hparams.tgt_vocab_size
		self.share_vocab = hparams.share_vocab
		self._init_embeddings()
		
	def _init_embeddings(self):
		self.embedding_encoder, self.embedding_decoder = \
			create_emb_for_encoder_and_decoder(
				src_vocab_size=self.src_vocab_size,
				tgt_vocab_size=self.tgt_vocab_size,
				embed_size=self.num_units,
				share_vocab=self.share_vocab)

	def _build_graph(self, hparams, scope=None):
		"""Creates a seq2seq model, however, all the sub_function must be implemented in the sub_class

		Args:
			hparams: Hyperparameter configurations
			scope: default 'dynamic_seq2seq'

		Returns:
		
		Raises:

		"""
		_info('Creating {} graph'.format(self.mode))

		with tf.variable_scope(scope or 'dynamic_seq2seq', dtype=self.dtype):
			# Encoder
			self.encoder_outputs, encoder_state = self._build_encoder(hparams)
			# Decoder
			logits, _, sample_id, final_context_state = (
				self._build_decoder(self.encoder_outputs, encoder_state, hparams))
			
			if self.mode != 'TRAIN':
				loss = self._compute_loss(logits)
			else:
				loss = tf.constant(0.0)
		
		return logits, loss, final_context_state, sample_id

	@abc.abstractmethod
	def _build_encoder(self, hparams):
		"""implemented by sub_class
		"""
		pass
	
	@abc.abstractmethod
	def _build_decoder(self, encoder_outputs, encoder_state, hparams):
		"""Build and run a RNN decoder

		Args:
			encoder_outputs: the outputs of encoder from each time step
			encoder_state: the final state of the encoder
			hparams: the hyperparameters configurations
		
		Return:
			a tuple of logits and final decoder state:
				logits: [batch_size, time, vocab_size]
		"""
		tgt_sos_id = tf.cast(hparams.sos, tf.int32)
		tgt_eos_id = tf.cast(hparams.eos, tf.int32)

		# get maximum decoder step
		maximum_iterations = self._get_infer_maximum_iterations(hparams)

		# Decoder
		with tf.variable_scope('decoder') as decoder_scope:
			self.output_layer = tf.layers.Dense(self.tgt_vocab_size, use_bias=False, name='output_projection')
			cell, decoder_initial_state = self._build_decoder_cell(
				hparams, encoder_outputs, encoder_state)

			logits = tf.no_op()
			decoder_cell_outputs = None

			if self.mode == 'TRAIN':
				# 1. embedding the decoder input
				decoder_emb_inp = tf.nn.embedding_lookup(
					self.embedding_decoder, self.decoder_input_data)
				# 2. build helper
				helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, self.seq_length_decoder_input_data)
				# 3. build decoder
				#TODO VAE here 
				my_decoder = tf.contrib.seq2seq.BasicDecoder(
					cell, helper, decoder_initial_state)
				# 4. decode now
				outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
					my_decoder, swap_memory=True, scope=decoder_scope)
				# get the max idx
				sample_id = outputs.sample_id
				# projection to the output layer
				decoder_cell_outputs = outputs.rnn_output
				logits = self.output_layer(decoder_cell_outputs)
			else:
				infer_mode = hparams.infer_mode
				start_tokens = tf.fill([self.batch_size], tgt_sos_id)
				end_token = tgt_eos_id
				_info('Decoder by mode= {}, beam_width={}, length_penalty={}, coverage_penalty={}'.format
					 (hparams.infer_mode, hparams.beam_width, hparams.length_penalty, hparams.coverage_penalty))
				
				if infer_mode == 'beam_search':
					# watch the difference between the 'beam_search' and 'greedy'
					# no need to build helper in 'beam_search'
					beam_width = hparams.beam_width
					length_panalty = hparams.length_panalty
					coverage_penalty = hparams.coverage_penalty

					my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
						cell=cell,
						embedding=self.embedding_decoder,
						start_tokens=start_tokens,
						end_token=end_token,
						initial_state=decoder_initial_state,
						beam_width=beam_width,
						output_layer=self.output_layer,
						length_panalty=length_panalty,
						coverage_penalty=coverage_penalty)
				elif infer_mode == 'sample':
					# TODO do it future
					pass
				elif infer_mode == 'greedy':
					helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
						self.embedding_decoder, start_tokens, end_token)
				else:
					_error('Unknonw infer model {}'.format(infer_mode))
					raise ValueError
				
				# if 'beam_search', my_decoder has been created
				if infer_mode != 'beam_search':
					my_decoder = tf.contrib.seq2seq.BasicDecoder(
						cell, helper, decoder_initial_state, output_layer=self.output_layer)
				
				outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
					my_decoder, maximum_iterations=maximum_iterations,
					swap_memory=True, scope=decoder_scope)
				
				if infer_mode == 'beam_search':
					sample_id = outputs.predicted_ids
				else:
					logits = outputs.rnn_output
					sample_id = outputs.sample_id
		
		return logits, decoder_cell_outputs, sample_id, final_context_state

	def _get_infer_maximum_iterations(self, hparams):
		"""Maximum decoding steps
		"""
		if hparams.max_length:
			maximum_iterations = int(hparams.max_length)
			_info('Decoding at maximum iterations {}'.format(maximum_iterations))
		else:
			decoding_length_factor = 2.0
			max_encoder_length = tf.reduce_max(self.seq_length_encoder_intput_data)
			maximum_iterations = tf.to_int32(tf.round(
				tf.to_float(max_encoder_length) * decoding_length_factor))
		return maximum_iterations
	
	@abc.abstractmethod
	def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state):
		"""Implemented by the sub_class

		Args:
			hparams: Hyperparameters configurations
			encoder_outputs: the outputs of encoder for each time step
			encoder_state: the final state of the encoder
		
		Returns:
			A tuple of a multi-layer RNN cell
		"""
		pass

	def _compute_loss(self, logits):
		# 1. compute the cross entropy loss
		cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logtis(
			labels=self.decoder_output_data, logits=logits)
		# 2. compute the mask
		max_time = tf.shape(self.decoder_output_data)[1]
		target_weights = tf.sequence_mask(
			self.seq_length_decoder_input_data, max_time, dtype=tf.float32)
		loss = tf.reduce_sum(cross_entropy_loss * target_weights) / tf.to_float(self.batch_size)
		
		#TODO add kl loss

		return loss

	def _get_lr_warmup(self, hparams):
		warmup_steps = hparams.warmup_steps
		warmup_scheme = hparams.warmup_scheme
		_info('learning_rate={:.2f}, warm_steps={}, warm_scheme={}' 
			.format(hparams.learning_rate, warmup_steps, warmup_scheme))
		
		if warmup_scheme == 't2t':
			warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
			inv_decay = warmup_factor ** (
				tf.to_float(warmup_steps - self.global_step))
		else:
			_error('Unknown warmup scheme {}'.format(warmup_scheme))
			raise ValueError
		
		return tf.cond(
			self.global_step < hparams.warmup_steps,
			lambda : inv_decay * self.learning_rate,
			lambda : self.learning_rate,
			name='learning_rate_warmup_cond')
	
	def _get_decay_info(self, hparams):
		decay_factor = 0.5
		start_decay_step = int(hparams.num_train_steps / 2)
		decay_times = 5
		remain_steps = hparams.num_train_steps - start_decay_step
		decay_steps = int(remain_steps / decay_times)
		return start_decay_step, decay_steps, decay_factor

	def _get_lr_decay(self, hparams):
		start_decay_step, decay_steps, decay_factor = self._get_decay_info(hparams)
		
		return tf.cond(
			self.global_step < start_decay_step,
			lambda : self.learning_rate,
			lambda : tf.train.exponential_decay(
				self.learning_rate,
				(self.global_step - start_decay_step),
				decay_steps, decay_factor, staircase=True),
			name='learning_rate_decay_cond')
	
	def _get_train_summary(self):
		train_summary = tf.summary.merge(
			[tf.summary.scalar('lr', self.learning_rate),
			tf.summary.scalar('train_loss', self.train_loss)])
		return train_summary
	
	def _get_infet_summary(self, hparams):
		del hparams
		return tf.no_op()
	
	def train(self, sess, realv):
		assert self.mode == 'TRAIN'
		feed = {self.encoder_input_data:realv[0],
				self.decoder_input_data:realv[1],
				self.decoder_output_data:realv[2],
				self.seq_length_encoder_intput_data:realv[3],
				self.seq_length_decoder_input_data:realv[4]}

		output_tuple = TrainOutputTuple(
			train_summary=self.train_summary,
			train_loss=self.train_loss,
			learning_rate=self.learning_rate)

		return sess.run([self.update, output_tuple], feed_dict=feed)
	
	def eval(self, sess, realv):
		assert self.mode == 'EVAL'
		feed = {self.encoder_input_data:realv[0],
				self.decoder_input_data:realv[1],
				self.decoder_output_data:realv[2],
				self.seq_length_encoder_intput_data:realv[3],
				self.seq_length_decoder_input_data:realv[4]}
		output_tuple = EvalOutputTuple(eval_loss=self.eval_loss)
		return sess.run(output_tuple, feed_dict=feed)
	
	def infer(self, sess, realv):
		assert self.mode == 'INFER'
		feed = {self.encoder_input_data:realv[0],
				self.seq_length_encoder_intput_data:realv[3]}
		output_tuple = InferOutputTuple(sample_id=self.sample_id)
		return sess.run(output_tuple, feed_dict=feed)