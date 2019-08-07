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
    """Base Abstract Model, 
    """
    def __init__(self, hparams, mode, scope=None):
        
        # TODO 1. parse the hparams
        # TODO 2. build the graph
        # TODO 3. Saver
        pass
        
    def _set_params_initializer(self):
    # TODO
        pass

if __name__ == '__main__':
    t = TrainOutputTuple(a='a', b='bb', c='ccc')
    print(t)