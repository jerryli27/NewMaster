# -*- coding: utf-8 -*-

##########################################################
#
# Attention-based Convolutional Neural Network
#   for Context-wise Learning
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
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


import util
import preprocessing_util

FLAGS = tf.app.flags.FLAGS


this_dir = os.path.abspath(os.path.dirname(__file__))

# eval parameters
tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'checkpoints'), 'Directory of the checkpoint files')
#tf.app.flags.DEFINE_boolean('save_fig', False, 'Whether save the visualized image or not')

CLASS_NAMES = ('A is-a B', 'B is-a A', 'Neither')

def evaluate(eval_data, config):
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

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(config['train_dir'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            #embeddings = sess.run(tf.all_variables())[0]

            print "\nStart evaluation\n"
            #losses = []
            #precision = []
            #recall = []
            #batches = util.batch_iter(eval_data, batch_size=config['batch_size'], num_epochs=1, shuffle=False)
            #for batch in batches:
            if config.has_key('contextwise') and config['contextwise']:
                left_batch, middle_batch, right_batch, y_batch, _ = zip(*eval_data)
                feed = {m.left: np.array(left_batch),
                        m.middle: np.array(middle_batch),
                        m.right: np.array(right_batch),
                        m.labels: np.array(y_batch)}
            else:
                x_batch, y_batch, _ = zip(*eval_data)
                feed = {m.inputs: np.array(x_batch), m.labels: np.array(y_batch)}
            loss, eval, actual_output, eval_per_class = sess.run([m.total_loss, m.eval_op, m.scores, m.eval_class_op], feed_dict=feed)
            #losses.append(loss)
            pre, rec = zip(*eval)
            #precision.append(pre)
            #recall.append(rec)

            avg_precision = np.mean(np.array(pre))
            avg_recall = np.mean(np.array(rec))
            auc = util.calc_auc_pr(pre, rec)
            f1 = (2.0 * pre[5] * rec[5]) / (pre[5] + rec[5])
            print '%s: Overall\nloss = %.6f, f1 = %.4f, auc = %.4f' % (datetime.now(), loss, f1, auc)

            pre_per_class, rec_per_class = zip(*eval_per_class)
            num_class = len(pre_per_class)
            for class_i in range(num_class):
                current_pre = pre_per_class[class_i]
                current_rec = rec_per_class[class_i]
                current_auc = util.calc_auc_pr(current_pre, current_rec)
                current_f1 = (2.0 * current_pre[5] * current_rec[5]) / (current_pre[5] + current_rec[5])
                print 'Class "%s": precision = %.4f, recall = %.4f, f1 = %.4f, auc = %.4f' % (CLASS_NAMES[class_i], current_pre[5], current_rec[5], current_f1, current_auc)


    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    plot_precision_recall(y_batch,actual_output)

    return pre, rec, x_batch, y_batch, actual_output

def plot_precision_recall(y_acutal_output, y_expected_output):
    # The majority of the code is taken from
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    assert len(y_acutal_output.shape) == 2

    n_classes = y_acutal_output.shape[1]

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_acutal_output[:, i],
                                                            y_expected_output[:, i])
        average_precision[i] = average_precision_score(y_acutal_output[:, i], y_expected_output[:, i])

    # Plot Precision-Recall curve

    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2
    # Plot Precision-Recall curve for each class
    plt.clf()
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(CLASS_NAMES[i], average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()



def main(argv=None):
    restore_param = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir

    source_path = os.path.join(restore_param['data_dir'], 'test_cs_unlabeled_data_combined.txt')
    target_path = os.path.join(restore_param['data_dir'], 'test_cs_labels_combined.txt')
    vocab_path = os.path.join(restore_param['data_dir'], 'test_cs_vocab_combined')
    _, data = util.read_data(source_path, target_path, restore_param['sent_len'],
                             train_size=restore_param['train_size'], hide_key_phrases=restore_param['hide_key_phrases'])

    pre, rec, x_input, expected_output, actual_output = evaluate(data, restore_param)

    actual_output_exp = np.exp(actual_output)
    actual_output_softmax = actual_output_exp / np.sum(actual_output_exp, axis=1, keepdims=True)

    output_difference = np.sum(np.abs(actual_output_softmax - expected_output), axis=1)


    sentence_indices_input = x_input[:,:-2]
    _,rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)
    sentence_input = preprocessing_util.indices_to_sentences(sentence_indices_input,rev_vocab)

    kp_indices_input = x_input[:,-2:]

    print('Diff\tType\tSentence\t\tExpected Score (A is-a B, B is-a A, Neither)\tActual Score')
    for sentence_i, sentence in enumerate(sentence_input):
        # Label the key phrases of interest in the current sentence with *.
        sentence[kp_indices_input[sentence_i,1]] += '*'
        sentence[kp_indices_input[sentence_i,0]] += '*'
        current_type = 'Neither'
        if expected_output[sentence_i,0] == 1:
            current_type = 'A is-a B'
        elif expected_output[sentence_i,1] == 1:
            current_type = 'B is-a A'

        print('%.3f\t%s\t%s\t\t%s\t%s\t'
              % (output_difference[sentence_i], current_type, ' '.join(sentence), str(expected_output[sentence_i]), str(actual_output_softmax[sentence_i])))

    util.dump_to_file(os.path.join(FLAGS.train_dir, 'results.cPickle'), {'precision': pre, 'recall': rec})


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
    tf.app.run()

