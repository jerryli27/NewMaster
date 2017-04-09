# -*- coding: utf-8 -*-

##########################################################
#
# Attention-based Convolutional Neural Network
#   for Context-wise Learning
# This file is for labeling unlabeled data.
#
#
#   Note: this implementation is mostly based on
#   https://github.com/may-/cnn-re-tf/blob/master/eval.py
#   https://github.com/yuhaozhang/sentence-convnet/blob/master/eval.py
#
##########################################################

from datetime import datetime
import os
import tensorflow as tf
import numpy as np


import util
import preprocessing_util



def label(eval_data, config):
    """ Build evaluation graph and run. """

    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            if config.has_key('contextwise') and config['contextwise']:
                import cnn_context
                m = cnn_context.Model(config, is_train=False)
            else:
                import cnn
                m = cnn.Model(config, is_train=False)
        saver = tf.train.Saver(tf.all_variables())

        tf_config = tf.ConfigProto()
        if config.get("gpu_percentage", 0) > 0:
            tf_config.gpu_options.per_process_gpu_memory_fraction = config.get("gpu_percentage", 0)
        else:
            tf_config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        with tf.Session(config=tf_config) as sess:
            ckpt = tf.train.get_checkpoint_state(config['train_dir'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            print "\nStart evaluation\n"

            x_data = np.array(eval_data)
            data_size = x_data.shape[0]
            batch_size = 10
            actual_output = []

            start_i = 0
            while start_i < data_size:
                end_i = min(start_i + batch_size, data_size)

                if config.has_key('contextwise') and config['contextwise']:
                    raise NotImplementedError
                    # left_batch, middle_batch, right_batch, y_batch, _ = zip(*eval_data)
                    # feed = {m.left: np.array(left_batch),
                    #         m.middle: np.array(middle_batch),
                    #         m.right: np.array(right_batch),
                    #         m.labels: np.array(y_batch)}
                else:
                    x_batch = eval_data[start_i:end_i]
                    feed = {m.inputs: x_batch}
                current_actual_output, = sess.run([m.scores], feed_dict=feed)
                actual_output.append(current_actual_output)
                start_i = end_i
    actual_output = np.concatenate(actual_output,axis=0)
    return x_data, actual_output



def main(argv=None):
    restore_param = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir

    source_path = os.path.join(restore_param['data_dir'], 'test_cs_unlabeled_data_combined.txt')
    target_path = os.path.join(restore_param['data_dir'], 'test_cs_labels_combined.txt')
    vocab_path = os.path.join(restore_param['data_dir'], 'test_cs_vocab_combined')
    data = util.read_data_unlabeled_part(source_path, target_path, restore_param['sent_len'])

    # # Now hard code to take the first 100
    # data = data[:100]
    # data = data[-1000:]

    x_input, actual_output = label(data, restore_param)

    actual_output_exp = np.exp(actual_output)
    actual_output_softmax = actual_output_exp / np.sum(actual_output_exp, axis=1, keepdims=True)
    actual_output_argmax = np.argmax(actual_output_softmax,axis=1)
    actual_output_softmax_sorted = np.argsort(-np.max(actual_output_softmax[...,:2], axis=1)).tolist()

    sentence_indices_input = x_input[:,:-2]
    _,rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)
    sentence_input = preprocessing_util.indices_to_sentences(sentence_indices_input,rev_vocab)

    kp_indices_input = x_input[:,-2:]

    print('Type\tSentence\t\tProbability [A is-a B, B is-a A, Neither]')
    # for sentence_i, sentence in enumerate(sentence_input):

    for sentence_i in actual_output_softmax_sorted:
        sentence = sentence_input[sentence_i]
        # Label the key phrases of interest in the current sentence with *.
        sentence[kp_indices_input[sentence_i,1]] += '*'
        sentence[kp_indices_input[sentence_i,0]] += '*'
        if actual_output_argmax[sentence_i] == 2:
            # current_type = 'Neither'
            break
        if actual_output_argmax[sentence_i] == 0:
            current_type = 'A is-a B'
        elif actual_output_argmax[sentence_i] == 1:
            current_type = 'B is-a A'

        print('%s\t%s\t\t%s\t'
              % (current_type, ' '.join(sentence), str(actual_output_softmax[sentence_i])))



    #if FLAGS.save_fig:
    #    import matplotlib.pyplot as plt
    #    plt.style.use('ggplot')

        # precision-recall curve
        #plt.plot(util.offset(rec, 0, 1), util.offset(pre, 1, 0))
    #    plt.plot(rec, pre)
    #    plt.title('Precision-Recall Curve')
    #    plt.xlabel('Recall')
    #    plt.ylabel('Precision')
    #    plt.savefig(os.path.join(FLAGS.train_dir, 'pr_curve.svg'))



if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS

    this_dir = os.path.abspath(os.path.dirname(__file__))

    # eval parameters
    tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'checkpoints'), 'Directory of the checkpoint files')
    # tf.app.flags.DEFINE_boolean('save_fig', False, 'Whether save the visualized image or not')
    tf.app.run()

