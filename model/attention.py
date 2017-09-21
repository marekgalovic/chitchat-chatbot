import tensorflow as tf
import numpy as np

from util import masked_softmax

class _Attention(object):
    
    def __init__(self, memory, memory_len=None):
        '''
        Abstract attention mechanism class.
        
        :param memory: Memory tensor to query. Shape: [num_steps x batch_size x embedding_size]
        :param memory_len: Optional memory tensor length. Shape: [batch_size]
        '''
        
        assert len(memory.get_shape()) == 3, 'Memory must have rank 3. [num_steps x batch_size x embedding_size]'
        assert memory.get_shape()[-1] is not None, 'Last dimension of memory can not be None'
        
        self._memory = memory
        self._memory_len = memory_len
        self._memory_width = int(self._memory.get_shape()[-1])
        
    def __call__(self, query):
        raise NotImplementedError
    
    def _align_query_with_memory(self, query):
        num_steps, _, _ = tf.unstack(tf.shape(self._memory))
        
        return tf.concat([
            tf.tile(tf.expand_dims(query, 0), [num_steps,1,1]),
            self._memory
        ], 2)

        
class BahdanauAttention(_Attention):
    
    def __init__(self, memory, mask_value=-np.inf, **kwargs):
        super(BahdanauAttention, self).__init__(memory, **kwargs)
        self._mask_value = mask_value
        
        with tf.name_scope('BahdanauAttention'):
            self._attention_W = tf.Variable(tf.truncated_normal([self._memory_width*2, 1], mean=0, stddev=0.1), dtype=tf.float32, name='weights')
            self._attention_b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='bias')
            
            tf.summary.histogram('attention_w', self._attention_W)
            tf.summary.histogram('attention_b', self._attention_b)
    
    def _attention_weights(self, query):
        num_steps, batch_size, _ = tf.unstack(tf.shape(self._memory))
        
        aligned = tf.reshape(self._align_query_with_memory(query), [-1, self._memory_width*2])
        weights = tf.reshape(tf.add(tf.matmul(aligned, self._attention_W), self._attention_b), [num_steps, batch_size, 1])
        
        if self._memory_len is not None:
            return masked_softmax(weights, self._memory_len, time_major=True, mask_value=self._mask_value)
    
        return tf.nn.softmax(weights, 0)
            
    def __call__(self, query):
        assert query.get_shape()[-1] == self._memory_width, 'Last dimension of query must have size %d' % (self._memory_width)
        
        weights = self._attention_weights(query)
        
        return tf.reduce_sum(tf.multiply(weights, self._memory), 0)
    

class GatedAttention(_Attention):
    
    def __init__(self, memory, residual=False, **kwargs):
        super(GatedAttention, self).__init__(memory, **kwargs)
        self._residual = residual
        
        with tf.name_scope('GatedAttention'):
            self._attention_W = tf.Variable(tf.truncated_normal([self._memory_width*2, self._memory_width], mean=0, stddev=0.1), dtype=tf.float32, name='weights')
            self._attention_b = tf.Variable(tf.zeros([self._memory_width]), dtype=tf.float32, name='bias')
            
            tf.summary.histogram('attention_w', self._attention_W)
            tf.summary.histogram('attention_b', self._attention_b)
            
    def _attention_weights(self, query):
        num_steps, batch_size, _ = tf.unstack(tf.shape(self._memory))
        
        aligned = tf.reshape(self._align_query_with_memory(query), [-1, self._memory_width*2])
        weights = tf.reshape(tf.add(tf.matmul(aligned, self._attention_W), self._attention_b), [num_steps, batch_size, self._memory_width])
        
        return tf.sigmoid(weights) 
    
    def __call__(self, query):
        assert query.get_shape()[-1] == self._memory_width, 'Last dimension of query must have size %d' % (self._memory_width)
        
        weights = self._attention_weights(query)
        c = tf.reduce_sum(tf.multiply(weights, self._memory), 0)
        
        if self._residual:
            c = tf.add(query, c)
        
        return c
