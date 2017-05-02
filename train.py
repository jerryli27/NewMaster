# -*- coding: utf-8 -*-

# Note this file is mostly took from https://github.com/may-/cnn-re-tf/blob/master/train.py

from datetime import datetime
import time
import os
import tensorflow as tf
import numpy as np

import cnn
import util

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



def train(train_data, test_data, FLAGS = tf.app.flags.FLAGS):
    # train_dir
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, timestamp))

    # save flags
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    FLAGS._parse_flags()
    config = dict(FLAGS.__flags.items())

    # Window_size must not be larger than the sent_len
    if config['sent_len'] < config['max_window']:
        config['max_window'] = config['sent_len']

    util.dump_to_file(os.path.join(out_dir, 'flags.cPickle'), config)


    num_batches_per_epoch = int(np.ceil(float(len(train_data))/FLAGS.batch_size))
    max_steps = num_batches_per_epoch * FLAGS.num_epochs



    with tf.Graph().as_default():
        with tf.variable_scope('cnn', reuse=None):
            m = cnn.Model(config, is_train=True)
        with tf.variable_scope('cnn', reuse=True):
            mtest = cnn.Model(config, is_train=False)

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
            train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "train"), graph=sess.graph)
            dev_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "dev"), graph=sess.graph)
            try:
                sess.run(tf.global_variables_initializer())
            except:
                sess.run(tf.initialize_all_variables())

            # assign pretrained embeddings
            if FLAGS.use_pretrain:
                print "Initialize model with pretrained embeddings..."
                print("Please don't forget to change the vocab size to the corresponding on in the embedding.")
                pretrained_embedding = np.load(os.path.join(FLAGS.data_dir, 'emb.npy'))
                m.assign_embedding(sess, pretrained_embedding)

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
                test_batches = util.batch_iter(test_data, batch_size=FLAGS.batch_size, num_epochs=1, shuffle=False)
                for batch in test_batches:
                    x_batch, y_batch, _ = zip(*batch)
                    #a_batch = np.ones((len(batch), 1), dtype=np.float32) / len(batch) # average
                    loss_value, eval_value = sess.run([mtest.total_loss, mtest.eval_op],
                        feed_dict={mtest.inputs: np.array(x_batch), mtest.labels: np.array(y_batch)})
                    dev_loss.append(loss_value)
                    pre, rec = zip(*eval_value)
                    dev_auc.append(util.calc_auc_pr(pre, rec))
                    dev_f1_score.append((2.0 * pre[5] * rec[5]) / (pre[5] + rec[5])) # threshold = 0.5

                return (np.mean(dev_loss), np.mean(dev_auc), np.mean(dev_f1_score))

            # train loop
            print "\nStart training (save checkpoints in %s)\n" % out_dir
            train_loss = []
            train_auc = []
            train_f1_score = []
            train_batches = util.batch_iter(train_data, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
            for batch in train_batches:
                batch_size = len(batch)

                m.assign_lr(sess, current_lr)
                global_step += 1

                x_batch, y_batch, a_batch = zip(*batch)
                feed = {m.inputs: np.array(x_batch), m.labels: np.array(y_batch)}
                if FLAGS.attention:
                    feed[m.attention] = np.array(a_batch)
                start_time = time.time()
                _, loss_value, eval_value = sess.run([m.train_op, m.total_loss, m.eval_op], feed_dict=feed)
                proc_duration = time.time() - start_time
                train_loss.append(loss_value)
                pre, rec = zip(*eval_value)
                auc = util.calc_auc_pr(pre, rec)
                f1 = (2.0 * pre[5] * rec[5]) / (pre[5] + rec[5]) # threshold = 0.5
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
    attention_path = None
    if FLAGS.attention:
        if os.path.exists(os.path.join(FLAGS.data_dir, 'source.att')):
            attention_path = os.path.join(FLAGS.data_dir, 'source.att')
        else:
            raise ValueError("Attention file %s not found.", os.path.join(FLAGS.data_dir, 'source.att'))
    train_data, test_data = util.read_data(source_path, target_path, FLAGS.sent_len,
                                           attention_path=attention_path, train_size=FLAGS.train_size,
                                           hide_key_phrases=FLAGS.hide_key_phrases)
    train(train_data, test_data)


if __name__ == '__main__':
    cnn.define_flags()
    define_flags()
    tf.app.run()
