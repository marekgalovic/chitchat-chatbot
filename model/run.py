import tensorflow as tf
import nltk
import json
import numpy as np

from seq2seq import Seq2Seq

word_index = json.load(open('./../../data/frames/word_index_new.json'))
dictionary = {}

for word, idx in word_index.items():
    dictionary[idx] = word

print('Dictionary size:', len(dictionary))

def parse_message(message):
    tokens = nltk.word_tokenize(str(message))
    
    return np.array([word_index[token] if token in word_index else word_index['<UNK>'] for token in tokens])

def pad_sequences(values, length, add_eos=False):
    result = []
    for row in values:
        row_value = list(row)
        if add_eos:
            row_value.append(1)
        if len(row_value) < length:
            row_value.extend([0] * (length - len(row_value)))
        if len(row_value) > length:
            if add_eos:
                row_value = row_value[:length-1] + [1]
            else:
                row_value = row_value[:length]
        result.append(row_value)
    return np.array(result)

def conversation_input(message, history_messages):
    history = [parse_message(m) for m in history_messages]
    history_size = [len(history)]
    history_seq_len = [len(m) for m in history]
    history = np.array([pad_sequences(history, max(map(len, history)))]).swapaxes(0,1)
    message = np.array([parse_message(message)]).T
    
    return message, [message.shape[0]], history, history_size, history_seq_len

def until_eos(tokens):
    for token in tokens:
        if token == 1:
            break
        yield token

with tf.Graph().as_default() as graph:
    input_seq = tf.placeholder(tf.int32, [None, None])
    target_seq = tf.placeholder(tf.int32, [None, None])
    input_seq_len = tf.placeholder(tf.int32, [None])
    target_seq_len = tf.placeholder(tf.int32, [None])
    history = tf.placeholder(tf.int32, [None, None, None])
    history_size = tf.placeholder(tf.int32, [None])
    history_seq_len = tf.placeholder(tf.int32, [None, None])

    model = Seq2Seq(600, [7800,300], feed_previous=True)
    model.graph(input_seq, target_seq, input_seq_len, target_seq_len, history, history_size, history_seq_len)

    with tf.Session(graph=graph) as sess:
        print('Runner:')
        print('---------------------------------------')
        # checkpoint_file = input('Checkpoint file: ')
        checkpoint_file = './../../models/gcloud_seq2seq_v7/3/model.ckpt-11494'
        tf.train.Saver().restore(sess, checkpoint_file)

        def response(message, history_messages):
            message, message_len, history, history_size, history_seq_len = conversation_input(message, history_messages)
            
            out = sess.run(model.decoder_embedding_ids, feed_dict={
                model._input_seq: message,
                model._input_seq_len: message_len,
                model._target_seq_len: np.array(message_len) + 20,
                model._history: history,
                model._history_size: history_size,
                model._history_seq_len: [history_seq_len]
            })

            out = list(until_eos(out.reshape(-1)))

            return ' '.join([dictionary[i] for i in out])

        history = ['<PAD>']
        print('Agent:', history[0])
        while True:
            user_sent = input('> ')
            resp = response(user_sent, history)
            print('Agent:', resp)
            history.append(user_sent)
            history.append(resp)

        print('Bye...')
