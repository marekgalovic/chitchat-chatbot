import tensorflow as tf
import numpy as np

def masked_softmax(values, lengths, time_major=False, mask_value=-np.inf):
    with tf.name_scope('MaskedSoftmax'):
        if time_major:
            mask = tf.expand_dims(tf.transpose(tf.sequence_mask(lengths, tf.reduce_max(lengths), dtype=tf.float32)), -1)
        else:
            mask = tf.expand_dims(tf.sequence_mask(lengths, tf.reduce_max(lengths), dtype=tf.float32), -2)

        inf_mask = (1 - mask) * mask_value
        inf_mask = tf.where(tf.is_nan(inf_mask), tf.zeros_like(inf_mask), inf_mask)

        return tf.nn.softmax(tf.multiply(values, mask) + inf_mask, 0 if time_major else -1)


class InputPipeline(object):
    
    def __init__(self, filenames, batch_size=32, n_epochs=50, capacity=1e4):
        self._filenames = filenames
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._capacity = int(capacity)
        
        print('InputPipeline - batch_size:', self._batch_size, 'n_epochs:', self._n_epochs, 'capacity:', self._capacity)
    

    def _queue_reader(self):
        with tf.device('/cpu:0'):
            file_queue = tf.train.string_input_producer(self._filenames, num_epochs = self._n_epochs, capacity=self._capacity)

            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(file_queue)

            return tf.parse_single_sequence_example(
                serialized_example,
                context_features = {
                    'input_seq': tf.VarLenFeature(tf.int64),
                    'target_seq': tf.VarLenFeature(tf.int64),
                    'input_seq_len': tf.FixedLenFeature([], tf.int64),
                    'target_seq_len': tf.FixedLenFeature([], tf.int64),
                    'history_size': tf.FixedLenFeature([], tf.int64),
                    'history_seq_len': tf.VarLenFeature(tf.int64),
                },
                sequence_features = {
                    'history': tf.VarLenFeature(tf.int64)
                }
            )
      

    def inputs(self):
        with tf.device('/cpu:0'):
            context_features, sequence_features = self._queue_reader()
            
            _, (input_seq, output_seq, input_seq_len, output_seq_len, history, history_size, history_seq_len) = tf.contrib.training.bucket_by_sequence_length(
                tf.reduce_mean([tf.cast(context_features['input_seq_len'], tf.int32), tf.cast(context_features['target_seq_len'], tf.int32)]),
                [
                    context_features['input_seq'],
                    context_features['target_seq'],
                    context_features['input_seq_len'],
                    context_features['target_seq_len'],
                    sequence_features['history'],
                    context_features['history_size'],
                    context_features['history_seq_len']
                ],
                batch_size = self._batch_size,
                bucket_boundaries=[5, 10, 15, 20, 30, 40, 50],
                num_threads=8,
                capacity = self._capacity,
                allow_smaller_final_batch = True,
                dynamic_pad=True
            )
            
            return (
                tf.transpose(tf.cast(tf.sparse_tensor_to_dense(input_seq), tf.int32)),
                tf.transpose(tf.cast(tf.sparse_tensor_to_dense(output_seq), tf.int32)),
                tf.cast(input_seq_len, tf.int32),
                tf.cast(output_seq_len, tf.int32),
                tf.transpose(tf.cast(tf.sparse_tensor_to_dense(history), tf.int32), [1,0,2]),
                tf.cast(history_size, tf.int32),
                tf.cast(tf.sparse_tensor_to_dense(history_seq_len), tf.int32),
            )
