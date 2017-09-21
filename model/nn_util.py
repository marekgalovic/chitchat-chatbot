import tensorflow as tf


class DenseLayer(object):
    
    def __init__(self, input_size, output_size, activation=None, name='DenseLayer'):
        self.input_size = input_size
        self.output_size = output_size
        self._activation = activation
        
        with tf.name_scope(name):
            self._W = tf.Variable(tf.truncated_normal([self.input_size, self.output_size], mean=0, stddev=0.1), dtype=tf.float32, name='W')
            self._b = tf.Variable(tf.zeros([self.output_size]), dtype=tf.float32, name='b')
            
            tf.summary.histogram('weights', self._W)
            tf.summary.histogram('bias', self._b)
    
    def __call__(self, inputs):
        linear = tf.add(tf.matmul(inputs, self._W), self._b)
        
        if self._activation is not None:
            return self._activation(linear)
        
        return linear
