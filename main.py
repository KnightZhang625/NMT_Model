# coding:utf-8
# Produced by Andysin Zhang
# 12_Aug_2019

import os
import math
import tensorflow as tf

from base_model import BaseModel
from utils.log import log_info as _info
from utils.log import log_error as _error
import model_helper as _mh

def select_model_creator(hparams):
	"""select the model object to create the model"""
	if hparams.model_type == 'standard':
		return BaseModel
	elif hparams.model_type == 'advanced' :
		# TODO create Attention model
		raise NotImplementedError
	else:
		_error('Unknown model type : {}'.format(hparams.model_type))
		raise ValueError

def statistic(loss, predict_count, batch_size, stats):
	stats['loss'] += loss * batch_size
	stats['count'] += predict_count

def reset_statistic(stats):
	stats['loss'] = 0.
	stats['count'] = 0
	
def train(hparams, datas, scope=None):
	"""build the train process"""
	
	# 1. create the model
	model_creator = select_model_creator(hparams)
	train_model = _mh.create_model(model_creator, hparams, 'train')
	# eval_model = _mh.create_model(model_creator, hparams, 'eval')
	# infer_model = _mh.create_model(model_creator, hparams, 'infer')

	# 2. create the session
	sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
	sess_conf.gpu_options.allow_growth = True
	train_sess = tf.Session(config=sess_conf, graph=train_model.graph)
	# eval_sess = tf.Session(config=sess_conf, graph=eval_model.graph)
	# infer_sess = tf.Session(config=sess_conf, graph=infer_model.graph)

	with train_model.graph.as_default():
		loaded_train_model, global_step = _mh.create_or_load_model(
			train_model.model, hparams.out_dir, train_sess)
	
	# Summary writer
	summary_writer = tf.summary.FileWriter(
		os.path.join(hparams.out_dir, hparams.summary_name), train_model.graph)
	
	num_steps = hparams.train_steps
	stats = {'loss': 0., 'count': 0}
	batch_size = hparams.batch_size
	while global_step < num_steps:
		# TODO iterate on the dataset, each global_step is the batch_size
		# when loop a batch, update the global_step,  could obtain from the model
		# datas = None
		feed_dict = datas
		res = loaded_train_model.train(train_sess, feed_dict)
		_info('STEP : {} \n\t LOSS : {:2f} LOSS_PER : {:2f} KL_LOSS : {:2f}'.format(
			global_step, res[1].train_loss, res[1].loss_per_token, res[1].kl_loss))
		global_step = res[1].global_step
		
		# update statistic
		statistic(res[1].train_loss, res[1].predict_count, batch_size, stats)

		if global_step % 100 == 0:
			try:
				ppl = math.exp(stats['loss'] / stats['count'])
			except OverflowError:
				ppl = float('inf')
			finally:
				_info('Perplexity : {:2f}'.format(ppl))
				reset_statistic(stats)
		
		if global_step % hparams.save_batch == 0:
			loaded_train_model.saver.save(
				train_sess,
				os.path.join(hparams.out_dir, 'nmt.ckpt'),
				global_step=global_step)
			_info('Save model at step {}th'.format(global_step))
	
	summary_writer.close()

if __name__ == '__main__':
	import numpy as np
	from hparameters import hparams

	input_x = np.array([[10, 120, 30, 0, 0], [20, 30, 0, 0, 0], [15, 20, 30, 50, 100]])
	seq_input_x = [3, 2, 5]
	output_y_input = np.array([[1, 20, 10, 30, 0, 0, 0], [1, 3, 3, 4, 5, 6, 7], [1, 20, 30, 0, 0, 0, 0]])
	output_y_output = np.array([[20, 10, 30, 2, 0, 0, 0], [3, 3, 4, 5, 6, 7, 2], [20, 30, 2, 0, 0, 0, 0]])
	seq_output_y = [4, 7, 3]
	
	datas = [input_x, output_y_input, output_y_output, seq_input_x, seq_output_y]
	train(hparams, datas)
	
	# # Infer test
	# model_creator = select_model_creator(hparams)
	# infer_model = _mh.create_model(model_creator, hparams, 'infer')
	# sess_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
	# sess_conf.gpu_options.allow_growth = True
	# infer_sess = tf.Session(config=sess_conf, graph=infer_model.graph)
	
	# data = [input_x, seq_input_x]
	# with infer_model.graph.as_default():
	# 	loaded_infer_model, global_step = _mh.create_or_load_model(
	# 		infer_model.model, hparams.out_dir, infer_sess)
	
	# res = loaded_infer_model.infer(infer_sess, data)
	# print(res.sample_id)
