import tensorflow as tf

from attention import _Attention
from nn_util import DenseLayer


class DeviceWrapper(tf.contrib.rnn.RNNCell):
    
    def __init__(self, cell, device):
        self._cell = cell
        self._device = device
        
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__+'ZeroState', values=[batch_size]):
            with tf.device(self._device):
                return self._cell.zero_state(batch_size, dtype)
    
    def __call__(self, inputs, state, scope=None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope=scope)
        

class ResidualWrapper(tf.contrib.rnn.RNNCell):
    
    def __init__(self, cell):
        self._cell = cell
    
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__+'ZeroState', values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)
    
    def __call__(self, inputs, state, scope=None):
        outputs, new_state = self._cell(inputs, state, scope=scope)
        
        return (inputs + outputs, new_state)


class AttentionWrapper(tf.contrib.rnn.RNNCell):
    
    def __init__(self, cell, attention):
        assert isinstance(attention, _Attention), 'Param :attention should be an _Attention class instance'
        
        with tf.name_scope('AttentionWrapper'):
            self._cell = cell
            self._attention = attention
            self._candidate_state = DenseLayer(cell.state_size*3, cell.state_size, activation=tf.tanh, name='candidate_state')
    
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__+'ZeroState', values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)
    
    def __call__(self, inputs, state, scope=None):
        new_output, new_state = self._cell(inputs, state, scope=scope)
        
        context = self._attention(new_state)
        attended_state = self._candidate_state(tf.concat([new_state, new_output, context], 1))
        
        return (new_output, attended_state)


class MultiAttentionWrapper(tf.contrib.rnn.RNNCell):
    
    def __init__(self, cell, *attentions):
        assert all([isinstance(attention, _Attention) for attention in attentions]), 'Param :attentions should be an _Attention class instance'
        
        with tf.name_scope('MultiAttentionWraper'):
            self._cell = cell
            self._attentions = attentions
            self._candidate_state = DenseLayer(cell.state_size*(2+len(self._attentions)), cell.state_size, activation=tf.tanh, name='candidate_state')
            
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size
    
    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__+'ZeroState', values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)
        
    def __call__(self, inputs, state, scope=None):
        new_output, new_state = self._cell(inputs, state, scope=scope)
        
        contexts = tf.concat([attention(new_state) for attention in self._attentions], 1)
        attended_state = self._candidate_state(tf.concat([new_state, new_output, contexts], 1))
        
        return (new_output, attended_state)
