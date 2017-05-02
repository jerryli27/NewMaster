# -*- coding: utf-8 -*-

##########################################################
#
# Attention-based Convolutional Neural Network
#   for Multi-label Multi-instance Learning
#
#
#   Note: this implementation is mostly based on
#   https://github.com/may-/cnn-re-tf/blob/master/cnn.py
#   https://github.com/yuhaozhang/sentence-convnet/blob/master/model.py
#
##########################################################

import tensorflow as tf

import neural_util

def define_flags():
    # model parameters
    tf.app.flags.DEFINE_integer('batch_size', 100, 'Training batch size')
    tf.app.flags.DEFINE_integer('word_emb_size', 300, 'Size of word embeddings')
    tf.app.flags.DEFINE_integer('pos_emb_size', 50, 'Size of positional embeddings')
    tf.app.flags.DEFINE_integer('num_hidden_layers', 1, 'Number of filters for each window size')
    tf.app.flags.DEFINE_integer('hidden_layers_size', 512, 'Number of filters for each window size')
    tf.app.flags.DEFINE_integer('vocab_size', 40000, 'Vocabulary size')
    tf.app.flags.DEFINE_integer('num_classes', 3, 'Number of class to consider')
    tf.app.flags.DEFINE_integer('sent_len', 128, 'Input sentence length.')
    tf.app.flags.DEFINE_float('l2_reg', 1e-4, 'l2 regularization weight')
    tf.app.flags.DEFINE_boolean('multi_label', False, 'Multilabel or not')

def define_new_flags():
    tf.app.flags.DEFINE_integer('num_hidden_layers', 1, 'Number of filters for each window size')
    tf.app.flags.DEFINE_integer('hidden_layers_size', 512, 'Number of filters for each window size')


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_on_gpu(name, shape, initializer):
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, initializer, wd, use_gpu=False):
    var = _variable_on_cpu(name, shape, initializer) if not use_gpu else _variable_on_gpu(name, shape, initializer)
    if wd is not None and wd != 0.:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    else:
        weight_decay = tf.constant(0.0, dtype=tf.float32)
    return var, weight_decay

def _auc_pr(true, prob, threshold):
    pred = tf.select(prob > threshold, tf.ones_like(prob), tf.zeros_like(prob))
    tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(true, tf.bool))
    fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(true, tf.bool)))
    fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(true, tf.bool))
    pre = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)))
    rec = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)), tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)))
    return pre, rec




class Model(object):

    def __init__(self, config, is_train=True):
        self.is_train = is_train
        self.word_emb_size = config['word_emb_size']
        self.pos_emb_size = config['pos_emb_size']
        self.batch_size = config['batch_size']
        self.vocab_size = config['vocab_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.hidden_layers_size = config['hidden_layers_size']
        self.num_classes = config['num_classes']
        self.sent_len = config['sent_len']
        self.l2_reg = config['l2_reg']
        self.multi_label = config['multi_label']
        if is_train:
            self.optimizer = config['optimizer']
            self.dropout = config['dropout']
        if config.get("gpu_percentage",0) > 0:
            self.gpu_percentage = config["gpu_percentage"]
            self.use_gpu = True
            # TODO: still need to change where the variable is located.
        else:
            self.gpu_percentage = 0
            self.use_gpu = False
        if config['hide_key_phrases']:
            self.hide_key_phrases = True
        else:
            self.hide_key_phrases = False

        self.build_graph()

    def build_graph(self):
        """ Build the computation graph. """
        self._inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.word_emb_size * 2], name='input_x')
        # self._sentences = self._inputs[:,:-2]
        # # Number of key phrases is always 2
        # self._key_phrase_indices = self._inputs[:,-2:]
        self._labels = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_y')
        losses = []

        # fully-connected layer
        prev_layer_size = 2 * self.word_emb_size
        prev_layer = self._inputs
        for i in range(self.num_hidden_layers):
            with tf.variable_scope('hidden_layer_%d'%(i)) as scope:
                W, wd = _variable_with_weight_decay('W', shape=[prev_layer_size, self.hidden_layers_size],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.05),
                                                    wd=self.l2_reg, use_gpu=self.use_gpu)
                losses.append(wd)
                biases = _variable_on_cpu('bias', shape=[self.hidden_layers_size],
                                          initializer=tf.constant_initializer(0.01)) if not self.use_gpu else _variable_on_gpu(
                    'bias', shape=[self.hidden_layers_size],
                    initializer=tf.constant_initializer(0.01))
                current_layer = tf.nn.bias_add(tf.matmul(prev_layer, W), biases, name='hidden_layer_%d'%(i))
                prev_layer = current_layer
                prev_layer_size = self.hidden_layers_size


        with tf.variable_scope('output') as scope:
            W, wd = _variable_with_weight_decay('W', shape=[prev_layer_size, self.num_classes],
                                                initializer=tf.truncated_normal_initializer(stddev=0.05),
                                                wd=self.l2_reg, use_gpu=self.use_gpu)
            losses.append(wd)
            biases = _variable_on_cpu('bias', shape=[self.num_classes],
                                      initializer=tf.constant_initializer(0.01)) if not self.use_gpu else _variable_on_gpu('bias', shape=[self.num_classes],
                                      initializer=tf.constant_initializer(0.01))
            self.logits = tf.nn.bias_add(tf.matmul(prev_layer, W), biases, name='logits')

        # loss
        with tf.variable_scope('loss') as scope:
            if self.multi_label:
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self._labels,
                                                                        name='cross_entropy_per_example')
            else:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, self._labels,
                                                                        name='cross_entropy_per_example')


            cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')

            losses.append(cross_entropy_loss)
            self._total_loss = tf.add_n(losses, name='total_loss')


        # eval with precision-recall
        with tf.variable_scope('evaluation') as scope:
            precision = []
            recall = []
            for threshold in range(10, -1, -1):
                pre, rec = _auc_pr(self._labels, self.logits, threshold * 0.1)
                precision.append(pre)
                recall.append(rec)
            self._eval_op = zip(precision, recall)

            # f1 score on threshold=0.5
            #self._f1_score = tf.truediv(tf.mul(tf.constant(2.0, dtype=tf.float64),
            #                                 tf.mul(precision[5], recall[5])), tf.add(precision, recall))

            labels_per_class = tf.unpack(self._labels,axis=1)
            logits_per_class = tf.unpack(self.logits,axis=1)
            precision = []
            recall = []
            for class_i in range(len(labels_per_class)):
                current_labels = labels_per_class[class_i]
                current_logits = logits_per_class[class_i]
                current_pre = []
                current_rec = []
                for threshold in range(10, -1, -1):
                    pre, rec = _auc_pr(current_labels, current_logits, threshold * 0.1)
                    current_pre.append(pre)
                    current_rec.append(rec)
                precision.append(current_pre)
                recall.append(current_rec)
            self._eval_class_op = zip(precision, recall)


        # train on a batch
        self._lr = tf.Variable(0.0, trainable=False)
        if self.is_train:
            if self.optimizer == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(self._lr)
            elif self.optimizer == 'adagrad':
                opt = tf.train.AdagradOptimizer(self._lr)
            elif self.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(self._lr)
            elif self.optimizer == 'sgd':
                opt = tf.train.GradientDescentOptimizer(self._lr)
            else:
                raise ValueError("Optimizer not supported.")
            grads = opt.compute_gradients(self._total_loss)
            self._train_op = opt.apply_gradients(grads)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
        else:
            self._train_op = tf.no_op()

        return

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def eval_op(self):
        return self._eval_op

    @property
    def eval_class_op(self):
        return self._eval_class_op

    @property
    def scores(self):
        return self.logits
    #
    # @property
    # def W_emb(self):
    #     return self._W_emb

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    # def assign_embedding(self, session, pretrained):
    #     session.run(tf.assign(self.W_emb, pretrained))
