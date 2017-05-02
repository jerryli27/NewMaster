# -*- coding: utf-8 -*-

# This file uses SVM to classify pairs of key phrases extracted from CS papers on whether there exists a sub-category
# relation between the two.

from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np
from sklearn import svm

import util
import multilayer


def define_flags():
    # train parameters
    this_dir = os.path.abspath(os.path.dirname(__file__))
    tf.app.flags.DEFINE_string('data_dir', os.path.join(this_dir, 'hand_label_context_tool'), 'Directory of the data')
    tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'checkpoints'),
                               'Directory to save training checkpoint files')
    tf.app.flags.DEFINE_integer('train_size', 100000, 'Number of training examples')
    tf.app.flags.DEFINE_integer('num_epochs', 20, 'Number of epochs to run')
    tf.app.flags.DEFINE_boolean('use_pretrain', False, 'Use word2vec pretrained embeddings or not')
    tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether log device information in summary')

    tf.app.flags.DEFINE_string('optimizer', 'adam',
                               'Optimizer to use. Must be one of "sgd", "adagrad", "adadelta" and "adam"')
    tf.app.flags.DEFINE_float('init_lr', 0.01, 'Initial learning rate')
    tf.app.flags.DEFINE_float('lr_decay', 0.95, 'LR decay rate')
    tf.app.flags.DEFINE_integer('tolerance_step', 500,
                                'Decay the lr after loss remains unchanged for this number of steps')
    tf.app.flags.DEFINE_float('dropout', 0.5, 'Dropout rate. 0 is no dropout.')
    tf.app.flags.DEFINE_boolean('hide_key_phrases', False, 'Whether to hide the key phrase pair in the input sentence by '
                                                           'replacing them with UNK.')

    # logging
    tf.app.flags.DEFINE_integer('log_step', 10, 'Display log to stdout after this step')
    tf.app.flags.DEFINE_integer('summary_step', 50,
                                'Write summary (evaluate model on dev set) after this step')
    tf.app.flags.DEFINE_integer('checkpoint_step', 100, 'Save model after this step')

    # Device option
    tf.app.flags.DEFINE_float('gpu_percentage', -1, "The percentage of gpu this program can use. "
                                                     "Set to <= 0 for cpu mode.")

def get_key_phrases(data):
    try:
        x, y, _ = zip(*data)
    except:
        x = data
    # x,y = data
    key_phrases = []
    for source_ids in x:
        key_phrases.append((source_ids[source_ids[len(source_ids) - 1]], source_ids[source_ids[len(source_ids) - 2]]))

    return np.array(key_phrases, dtype=np.int32)

def key_phrase_indices_to_embedding(key_phrase_indices, pretrained_embedding):
    num_pairs = key_phrase_indices.shape[0]
    embedding_dim = pretrained_embedding.shape[1]
    return np.reshape(pretrained_embedding[key_phrase_indices], (num_pairs, 2 * embedding_dim))

def _auc_pr(true, prob, threshold):
    pred = tf.select(prob > threshold, tf.ones_like(prob), tf.zeros_like(prob))
    tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(true, tf.bool))
    fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(true, tf.bool)))
    fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(true, tf.bool))
    pre = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)))
    rec = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)))
    return pre, rec



def train(train_data, test_data, FLAGS = tf.app.flags.FLAGS):
    # # train_dir
    # timestamp = str(int(time.time()))
    # out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, timestamp))
    #
    # # save flags
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # FLAGS._parse_flags()
    # config = dict(FLAGS.__flags.items())
    #
    # # Window_size must not be larger than the sent_len
    # if config['sent_len'] < config['max_window']:
    #     config['max_window'] = config['sent_len']
    #
    # util.dump_to_file(os.path.join(out_dir, 'flags.cPickle'), config)


    train_x = get_key_phrases(train_data)
    _, train_y, _ = zip(*train_data)

    test_x = get_key_phrases(test_data)
    _, test_y, _ = zip(*test_data)

    # # assign pretrained embeddings
    # if FLAGS.use_pretrain:
    print "Initialize model with pretrained embeddings..."
    print("Please don't forget to change the vocab size to the corresponding on in the embedding.")
    pretrained_embedding = np.load(os.path.join(FLAGS.data_dir, 'emb.npy'))
    train_x = key_phrase_indices_to_embedding(train_x, pretrained_embedding)
    test_x = key_phrase_indices_to_embedding(test_x, pretrained_embedding)

    # Use SVM. But SVM does not output a probability
    # train_y = np.argmax(train_y, axis=1)
    # test_y = np.argmax(test_y, axis=1)
    # clf = svm.SVC(class_weight='balanced')
    # clf.fit(train_x, train_y)
    # predicted_test_y = clf.predict(test_x)

    # Use fully connected multilayer nn.
    config = dict(FLAGS.__flags.items())
    # train_dir
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, timestamp))

    # save flags
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    FLAGS._parse_flags()
    config = dict(FLAGS.__flags.items())

    util.dump_to_file(os.path.join(out_dir, 'flags.cPickle'), config)


    num_batches_per_epoch = int(np.ceil(float(len(train_data))/FLAGS.batch_size))
    max_steps = num_batches_per_epoch * FLAGS.num_epochs

    with tf.Graph().as_default():
        with tf.variable_scope('multilayer', reuse=None):
            m = multilayer.Model(config, is_train=True)
        with tf.variable_scope('multilayer', reuse=True):
            mtest = multilayer.Model(config, is_train=False)
        # checkpoint
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        save_path = os.path.join(out_dir, 'model.ckpt')
        try:
            summary_op = tf.summary.merge_all()
        except:
            summary_op = tf.merge_all_summaries()

        # session
        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        if FLAGS.gpu_percentage > 0:
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_percentage
        else:
            config = tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement,
                device_count={'GPU': 0}
            )
        sess = tf.Session(config=config)
        with sess.as_default():
            train_summary_writer = tf.train.SummaryWriter(os.path.join(out_dir, "train"), graph=sess.graph)
            dev_summary_writer = tf.train.SummaryWriter(os.path.join(out_dir, "dev"), graph=sess.graph)
            try:
                sess.run(tf.global_variables_initializer())
            except:
                sess.run(tf.initialize_all_variables())

            # # assign pretrained embeddings
            # if FLAGS.use_pretrain:
            #     print "Initialize model with pretrained embeddings..."
            #     print("Please don't forget to change the vocab size to the corresponding on in the embedding.")
            #     pretrained_embedding = np.load(os.path.join(FLAGS.data_dir, 'emb.npy'))
            #     m.assign_embedding(sess, pretrained_embedding)

            # initialize parameters
            current_lr = FLAGS.init_lr
            lowest_loss_value = float("inf")
            decay_step_counter = 0
            global_step = 0

            # evaluate on dev set
            def dev_step(mtest, sess):
                dev_loss = []
                dev_auc = []
                dev_f1_score = []

                # create batch
                test_batches = util.batch_iter(zip(test_x, test_y), batch_size=FLAGS.batch_size, num_epochs=1, shuffle=False)
                for batch in test_batches:
                    x_batch, y_batch, = zip(*batch)
                    # a_batch = np.ones((len(batch), 1), dtype=np.float32) / len(batch) # average
                    loss_value, eval_value = sess.run([mtest.total_loss, mtest.eval_op],
                                                      feed_dict={mtest.inputs: np.array(x_batch),
                                                                 mtest.labels: np.array(y_batch)})
                    dev_loss.append(loss_value)
                    pre, rec = zip(*eval_value)
                    dev_auc.append(util.calc_auc_pr(pre, rec))
                    dev_f1_score.append((2.0 * pre[5] * rec[5]) / (pre[5] + rec[5]))  # threshold = 0.5

                return (np.mean(dev_loss), np.mean(dev_auc), np.mean(dev_f1_score))

            # train loop
            print "\nStart training (save checkpoints in %s)\n" % out_dir
            train_loss = []
            train_auc = []
            train_f1_score = []
            train_batches = util.batch_iter(zip(train_x, train_y), batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
            for batch in train_batches:
                batch_size = len(batch)

                m.assign_lr(sess, current_lr)
                global_step += 1

                x_batch, y_batch, = zip(*batch)
                feed = {m.inputs: np.array(x_batch), m.labels: np.array(y_batch)}
                start_time = time.time()
                _, loss_value, eval_value = sess.run([m.train_op, m.total_loss, m.eval_op], feed_dict=feed)
                proc_duration = time.time() - start_time
                train_loss.append(loss_value)
                pre, rec = zip(*eval_value)
                auc = util.calc_auc_pr(pre, rec)
                f1 = (2.0 * pre[5] * rec[5]) / (pre[5] + rec[5])  # threshold = 0.5
                train_auc.append(auc)
                train_f1_score.append(f1)

                assert not np.isnan(loss_value), "Model loss is NaN."

                # print log
                if global_step % FLAGS.log_step == 0:
                    examples_per_sec = batch_size / proc_duration
                    format_str = '%s: step %d/%d, f1 = %.4f, auc = %.4f, loss = %.4f ' + \
                                 '(%.1f examples/sec; %.3f sec/batch), lr: %.6f'
                    print format_str % (datetime.now(), global_step, max_steps, f1, auc, loss_value,
                                        examples_per_sec, proc_duration, current_lr)

                # write summary
                if global_step % FLAGS.summary_step == 0:
                    summary_str = sess.run(summary_op)
                    train_summary_writer.add_summary(summary_str, global_step)
                    dev_summary_writer.add_summary(summary_str, global_step)

                    # summary loss, f1
                    train_summary_writer.add_summary(
                        _summary_for_scalar('loss', np.mean(train_loss)), global_step=global_step)
                    train_summary_writer.add_summary(
                        _summary_for_scalar('auc', np.mean(train_auc)), global_step=global_step)
                    train_summary_writer.add_summary(
                        _summary_for_scalar('f1', np.mean(train_f1_score)), global_step=global_step)

                    dev_loss, dev_auc, dev_f1 = dev_step(mtest, sess)
                    dev_summary_writer.add_summary(
                        _summary_for_scalar('loss', dev_loss), global_step=global_step)
                    dev_summary_writer.add_summary(
                        _summary_for_scalar('auc', dev_auc), global_step=global_step)
                    dev_summary_writer.add_summary(
                        _summary_for_scalar('f1', dev_f1), global_step=global_step)

                    print "\n===== write summary ====="
                    print "%s: step %d/%d: train_loss = %.6f, train_auc = %.4f, train_f1 = %.4f" \
                          % (datetime.now(), global_step, max_steps,
                             np.mean(train_loss), np.mean(train_auc), np.mean(train_f1_score))
                    print "%s: step %d/%d:   dev_loss = %.6f,   dev_auc = %.4f,   dev_f1 = %.4f\n" \
                          % (datetime.now(), global_step, max_steps, dev_loss, dev_auc, dev_f1)

                    # reset container
                    train_loss = []
                    train_auc = []
                    train_f1_score = []

                # decay learning rate if necessary
                if loss_value < lowest_loss_value:
                    lowest_loss_value = loss_value
                    decay_step_counter = 0
                else:
                    decay_step_counter += 1
                if decay_step_counter >= FLAGS.tolerance_step:
                    current_lr *= FLAGS.lr_decay
                    print '%s: step %d/%d, Learning rate decays to %.5f' % \
                          (datetime.now(), global_step, max_steps, current_lr)
                    decay_step_counter = 0

                # stop learning if learning rate is too low
                if current_lr < 1e-5:
                    break

                # save checkpoint
                if global_step % FLAGS.checkpoint_step == 0:
                    saver.save(sess, save_path, global_step=global_step)
            saver.save(sess, save_path, global_step=global_step)

            # Lastly evaluate the test set
            loss_value, predicted_test_y_logits = sess.run([mtest.total_loss, mtest.scores],
                                              feed_dict={mtest.inputs: np.array(test_x),
                                                         mtest.labels: np.array(test_y)})

            predicted_test_y = np.argmax(predicted_test_y_logits, axis=1)
            test_y = np.argmax(test_y, axis=1)

    result = (predicted_test_y == test_y)
    accuracy = np.sum(result.astype(np.int32)) / float(result.shape[0])

    print("Overall %f%% answers were correct. " %(float(accuracy * 100)))

    epsilon = 0.00000001
    num_categories = 3
    true_positive_per_category = [np.bitwise_and(test_y==category_i, predicted_test_y==category_i) for category_i in range(num_categories)]
    false_positive_per_category = [np.bitwise_and(test_y!=category_i, predicted_test_y==category_i) for category_i in range(num_categories)]
    true_negative_per_category = [np.bitwise_and(test_y!=category_i, predicted_test_y!=category_i) for category_i in range(num_categories)]
    false_negative_per_category = [np.bitwise_and(test_y==category_i, predicted_test_y!=category_i) for category_i in range(num_categories)]
    precision_per_category = [np.sum(true_positive_per_category[category_i].astype(np.int32)) /
                              float(np.sum(true_positive_per_category[category_i].astype(np.int32)) +
                               np.sum(false_positive_per_category[category_i].astype(np.int32)) + epsilon)
                              for category_i in range(num_categories)]

    recall_per_category = [np.sum(true_positive_per_category[category_i].astype(np.int32)) /
                              float(np.sum(true_positive_per_category[category_i].astype(np.int32)) +
                               np.sum(false_negative_per_category[category_i].astype(np.int32)) + epsilon)
                              for category_i in range(num_categories)]

    f1_per_category = [2 / (1 / (precision_per_category[category_i] + epsilon) +
                            1 / (recall_per_category[category_i] + epsilon)) for category_i in range(num_categories)]


    for category_i in range(num_categories):

        print("Category %d has f1 score: %f, precision: %f, and recall %f"
              %(category_i, f1_per_category[category_i], precision_per_category[category_i], recall_per_category[category_i]))

    return test_x, predicted_test_y_logits



def label(eval_data, config):
    print(eval_data.shape)
    eval_x = get_key_phrases(eval_data)

    pretrained_embedding = np.load(os.path.join(config["data_dir"], 'emb.npy'))
    eval_x = key_phrase_indices_to_embedding(eval_x, pretrained_embedding)
    """ Build evaluation graph and run. """

    with tf.Graph().as_default():
        with tf.variable_scope('multilayer'):
            m = multilayer.Model(config, is_train=False)
        saver = tf.train.Saver(tf.global_variables())

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

            data_size = eval_x.shape[0]
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
                    x_batch = eval_x[start_i:end_i]
                    feed = {m.inputs: x_batch}
                current_actual_output, = sess.run([m.scores], feed_dict=feed)
                actual_output.append(current_actual_output)
                start_i = end_i
    actual_output = np.concatenate(actual_output,axis=0)
    return eval_data, actual_output

def _summary_for_scalar(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=float(value))])


def main(argv=None):
    FLAGS = tf.app.flags.FLAGS
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)

    # load dataset
    # source_path = os.path.join(FLAGS.data_dir, 'ids.txt')
    # target_path = os.path.join(FLAGS.data_dir, 'target.txt')
    source_path = os.path.join(FLAGS.data_dir, 'test_cs_unlabeled_data_combined.txt')
    target_path = os.path.join(FLAGS.data_dir, 'test_cs_labels_combined.txt')
    train_data, test_data = util.read_data(source_path, target_path, FLAGS.sent_len,
                                           attention_path=None, train_size=FLAGS.train_size,
                                           hide_key_phrases=FLAGS.hide_key_phrases)
    train(train_data, test_data)


if __name__ == '__main__':
    define_flags()
    multilayer.define_flags()
    tf.app.run()
