import os
import tensorflow as tf

import util

if __name__=='__main__':

    FLAGS = tf.app.flags.FLAGS

    this_dir = os.path.abspath(os.path.dirname(__file__))

    # eval parameters
    tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'checkpoints'), 'Directory of the checkpoint files')

    restore_param = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir

    source_path = os.path.join(restore_param['data_dir'], 'test_cs_unlabeled_data_combined.txt')
    target_path = os.path.join(restore_param['data_dir'], 'test_cs_labels_combined.txt')
    vocab_path = os.path.join(restore_param['data_dir'], 'test_cs_vocab_combined')
    util.modify_labeled_data(source_path,target_path,['solvers','cplex'],restore_param['sent_len'],vocab_path)
