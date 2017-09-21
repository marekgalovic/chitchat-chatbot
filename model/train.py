import tensorflow as tf
from optparse import OptionParser

from seq2seq import Seq2Seq
from util import InputPipeline

# Run options
parser = OptionParser()
parser.add_option('--data-dir', dest='data_dir')
parser.add_option('--job-dir', dest='output_dir')
parser.add_option('--run-name', dest='run_name')

options, _ = parser.parse_args()
print('Data:', options.data_dir, 'Output:', options.output_dir)

def train():
    with tf.Graph().as_default() as graph:
        model = Seq2Seq(600, [7800,300])
        queue = InputPipeline(['{0}/frames/conversations_new.tfrecords'.format(options.data_dir)], batch_size = 64, n_epochs = 1000, capacity=1e4)

        input_seq, target_seq, input_seq_len, target_seq_len, history, history_size, history_seq_len = queue.inputs()
        loss = model.graph(input_seq, target_seq, input_seq_len, target_seq_len, history, history_size, history_seq_len)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(model.loss, global_step=global_step)

        sv = tf.train.Supervisor(
            graph=graph,
            logdir='{0}/seq2seq_v7/{1}'.format(options.output_dir, options.run_name),
            saver=tf.train.Saver(max_to_keep=None),
            summary_op=tf.summary.merge_all(),
            global_step=global_step
        )

        with sv.managed_session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            while not sv.should_stop():
                sess.run(train_op)

if __name__ == '__main__':
    train()
