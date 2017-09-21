import tensorflow as tf

from nn_util import DenseLayer
from rnn_util import MultiAttentionWrapper
from attention import BahdanauAttention


class Seq2Seq(object):
    
    def __init__(self, state_size, embeddings_shape, feed_previous=False):
        self._state_size = state_size
        self._embeddings_shape = embeddings_shape
        self._feed_previous = feed_previous
        
        print('Seq2Seq - stacked:')
        print('State size:', self._state_size)
        print('Embeddings shape:', self._embeddings_shape)
        print('Feed previous:', self._feed_previous)
        
    def graph(self, input_seq, target_seq, input_seq_len, target_seq_len, history, history_size, history_seq_len):
        '''
        :param input_seq: Input sequence tensor
        :param target_seq: Target sequence tensor
        :param input_seq_len: Input sequence length tensor
        :param target_seq_len: Target sequence length tensor
        :param history: Sequence history tensor
        :param history_size: Tensor representing number of sequences in history.
        :param history_seq_len: History sequences length
        '''
        self._input_seq = input_seq
        self._target_seq = target_seq
        self._input_seq_len = input_seq_len
        self._target_seq_len = target_seq_len
        # History
        self._history = history
        self._history_size = history_size
        self._history_seq_len = history_seq_len
        
        # Build
        self._initialize_embeddings()
        self._context_encoder()
        self._encoder()
        self._context_attention()
        self._decoder()
        
        return self._loss()
    
    def _initialize_embeddings(self):
        self._embeddings = tf.Variable(tf.random_uniform(self._embeddings_shape, -0.5, 0.5), dtype=tf.float32, name='embeddings')
        
        with tf.device('/cpu:0'):
            _, batch_size = tf.unstack(tf.shape(self._input_seq))
            
            self._input_seq_embedded = tf.nn.embedding_lookup(self._embeddings, self._input_seq)
            self._history_embbedded = tf.nn.embedding_lookup(self._embeddings, self._history)
            self._pad = tf.nn.embedding_lookup(self._embeddings, tf.zeros([batch_size], dtype=tf.int32))
            self._eos = tf.nn.embedding_lookup(self._embeddings, tf.ones([batch_size], dtype=tf.int32))
    
    def _context_encoder(self):
        # _context shape: [num_sentences x batch_size x embedding_size]
        with tf.variable_scope('context_encoder'):
            n_messages, batch_size, num_steps, _ = tf.unstack(tf.shape(self._history_embbedded))
            history_flattened = tf.transpose(tf.reshape(self._history_embbedded, [-1, num_steps, self._embeddings_shape[1]]), [1,0,2])
            
            fw_cell = tf.contrib.rnn.GRUCell(self._state_size)
            bw_cell = tf.contrib.rnn.GRUCell(self._state_size)
            
            _, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell,
                history_flattened,
                sequence_length=tf.reshape(self._history_seq_len, [-1]),
                dtype=tf.float32,
                time_major=True
            )
            
            # Project [2*state_size] -> [state_size]
            projected_state = tf.layers.dense(tf.concat(encoder_state, 1), self._state_size, activation=tf.tanh, name='context_encoder_state_projection')
            
            self._context = tf.reshape(projected_state, [n_messages, batch_size, self._state_size])
            tf.summary.histogram('context', self._context)
        
    def _encoder(self):
        with tf.variable_scope('encoder'):
            with tf.name_scope('bi-directional'):
                fw_cell = tf.contrib.rnn.GRUCell(self._state_size)
                bw_cell = tf.contrib.rnn.GRUCell(self._state_size)

                bi_encoder_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell, bw_cell,
                    self._input_seq_embedded,
                    sequence_length = self._input_seq_len,
                    dtype=tf.float32,
                    time_major=True
                )
                
                bi_encoder_outputs = tf.concat(bi_encoder_outputs, 2)
            
            with tf.name_scope('uni-directional'):
                cell = tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.GRUCell(self._state_size),
                    tf.contrib.rnn.GRUCell(self._state_size),
                ], state_is_tuple=True)
                
                uni_encoder_outputs, uni_encoder_state = tf.nn.dynamic_rnn(
                    cell,
                    bi_encoder_outputs,
                    sequence_length = self._input_seq_len,
                    dtype = tf.float32,
                    time_major = True
                )
                
                self._encoder_outputs = uni_encoder_outputs
                self._encoder_state = uni_encoder_state[-1]
            
            tf.summary.histogram('encoder_outputs', self._encoder_outputs)
            tf.summary.histogram('encoder_state', self._encoder_state)
    
    def _context_attention(self):
        with tf.name_scope('context_attention'):
            context_attention = BahdanauAttention(self._context, memory_len=self._history_size, mask_value=1e-18)

            self._encoder_state_with_context = context_attention(self._encoder_state)
        
    def _initialize_decoder_params(self):
        self._output_projection_layer = DenseLayer(self._state_size, self._embeddings_shape[0], name='output_projection')
        
        # Prepare targets tensor array if feed_previous=False
        if not self._feed_previous:
            with tf.device('/cpu:0'):
                _target_seq_embedded = tf.nn.embedding_lookup(self._embeddings, self._target_seq)
        
            _targets_ta = tf.TensorArray(dtype=tf.float32, size=tf.reduce_max(self._target_seq_len))
            self._targets_ta = _targets_ta.unstack(_target_seq_embedded)
    
    def _decoder(self):  
        with tf.variable_scope('decoder'):
            self._initialize_decoder_params()
            
            attention_cell = MultiAttentionWrapper(
                tf.contrib.rnn.GRUCell(self._state_size),
                BahdanauAttention(self._encoder_outputs, memory_len=self._input_seq_len),
                BahdanauAttention(self._context, memory_len=self._history_size, mask_value=1e-18)
            )

            cell = tf.contrib.rnn.MultiRNNCell([
                # Only first cell has attention
                attention_cell,
                # Other cells
                tf.contrib.rnn.GRUCell(self._state_size),
                tf.contrib.rnn.GRUCell(self._state_size)
            ], state_is_tuple=True)

            decoder_outputs_ta, _, _ = tf.nn.raw_rnn(cell, self._decoder_loop_fn)
            decoder_outputs = decoder_outputs_ta.stack()
            
            tf.summary.histogram('decoder_outputs', decoder_outputs)

            num_steps, batch_size, decoder_output_size = tf.unstack(tf.shape(decoder_outputs))
            self._decoder_logits = tf.reshape(
                self._output_projection_layer(tf.reshape(decoder_outputs, [-1, decoder_output_size])),
                [num_steps, batch_size, self._embeddings_shape[0]]
            )

            self.decoder_embedding_ids = tf.cast(tf.argmax(self._decoder_logits, 2), tf.int32)
    
    def _decoder_loop_fn(self, time, previous_output, previous_state, previous_loop_state):
        is_finished = tf.greater_equal(time, self._target_seq_len)
            
        if previous_state is None:
            # Initial state
            return (is_finished, self._eos, tuple([self._encoder_state_with_context]*3), None, None)
        
        def _next_input():
            if not self._feed_previous:
                return self._targets_ta.read(time - 1)
            
            embedding_ids = tf.argmax(self._output_projection_layer(previous_output), 1)
            with tf.device('/cpu:0'):
                return tf.nn.embedding_lookup(self._embeddings, embedding_ids)
        
        next_input = tf.cond(tf.reduce_all(is_finished), lambda: self._pad, _next_input)
        
        return (is_finished, next_input, previous_state, previous_output, None)
    
    def _loss(self):
        stepwise_ce = tf.nn.softmax_cross_entropy_with_logits(
            labels = tf.one_hot(self._target_seq, self._embeddings_shape[0]),
            logits = self._decoder_logits
        )
        
        self.loss = tf.reduce_mean(stepwise_ce)
        tf.summary.scalar('loss', self.loss)
        
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self._target_seq, self.decoder_embedding_ids), tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        
        return self.loss
