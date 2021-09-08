from inits import *
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class SpGAT_Conv(Layer):
    """Graph convolution layer."""
    def __init__(self, k_por, node_num,weight_normalize,input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(SpGAT_Conv, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.k_por = k_por
        self.node_num = node_num
        self.weight_normalize = weight_normalize
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_' + str(0)] = glorot([input_dim, output_dim],
                                                    name='weights_' + str(0))
            k_fre = int(self.k_por * self.node_num)
            init_alpha = np.array([1, 1], dtype='float32')
            self.alpha = tf.get_variable("tf_var_initialized_from_alpha", initializer = init_alpha, trainable=True)
            self.alpha = tf.nn.softmax(self.alpha) 
            self.vars['low_w'] = self.alpha[0]
            self.vars['high_w'] = self.alpha[1]


            self.vars['kernel_low'] = ones_fix([k_fre], name='kernel_low')
            self.vars['kernel_high'] = ones_fix([self.node_num - k_fre], name='kernel_high')
            self.vars['kernel_low'] = self.vars['kernel_low'] * self.vars['low_w']
            self.vars['kernel_high'] = self.vars['kernel_high'] * self.vars['high_w']


            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        supports_low = tf.matmul(tf.sparse_tensor_to_dense(self.support[0]),tf.diag(self.vars['kernel_low']),a_is_sparse=True,b_is_sparse=True)
        supports_low = tf.matmul(supports_low,tf.sparse_tensor_to_dense(self.support[1]),a_is_sparse=True,b_is_sparse=True)
        pre_sup = dot(x, self.vars['weights_' + str(0)],sparse=self.sparse_inputs)
        output_low = dot(supports_low,pre_sup)


        supports_high = tf.matmul(tf.sparse_tensor_to_dense(self.support[2]),tf.diag(self.vars['kernel_high']),a_is_sparse=True,b_is_sparse=True)
        supports_high = tf.matmul(supports_high,tf.sparse_tensor_to_dense(self.support[3]),a_is_sparse=True,b_is_sparse=True)
        output_high = dot(supports_high,pre_sup)
        
        #Mean Pooling
        #output = output_low + output_high
        #Max Pooling
        output = tf.concat([tf.expand_dims(output_low, axis = 0), tf.expand_dims(output_high, axis = 0)], axis = 0)
        output = tf.reduce_max(output, axis = 0)
        #import pdb; pdb.set_trace()
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
