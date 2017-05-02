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
    tf.app.flags.DEFINE_integer('num_kernel', 100, 'Number of filters for each window size')
    tf.app.flags.DEFINE_integer('min_window', 3, 'Minimum size of filter window')
    tf.app.flags.DEFINE_integer('max_window', 5, 'Maximum size of filter window')
    tf.app.flags.DEFINE_integer('vocab_size', 40000, 'Vocabulary size')
    tf.app.flags.DEFINE_integer('num_classes', 3, 'Number of class to consider')
    tf.app.flags.DEFINE_integer('sent_len', 128, 'Input sentence length.')
    tf.app.flags.DEFINE_float('l2_reg', 1e-4, 'l2 regularization weight')
    tf.app.flags.DEFINE_boolean('attention', False, 'Whether use attention or not')
    tf.app.flags.DEFINE_boolean('multi_label', False, 'Multilabel or not')




def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_on_gpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, initializer, wd, use_gpu = False):
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
        self.num_kernel = config['num_kernel']
        self.min_window = config['min_window']
        self.max_window = config['max_window']
        self.vocab_size = config['vocab_size']
        self.num_classes = config['num_classes']
        self.sent_len = config['sent_len']
        self.l2_reg = config['l2_reg']
        self.multi_instance = config['attention']
        self.multi_label = config['multi_label']
        if is_train:
            self.optimizer = config['optimizer']
            self.dropout = config['dropout']
        if config.get("gpu_percentage",0) > 0:
            self.gpu_percentage = config["gpu_percentage"]
            self.use_gpu = True
        else:
            # TODO: need to change where the variable is located. All variables are now located on cpu.
            self.gpu_percentage = 0
            self.use_gpu = False
        if config.get('hide_key_phrases',False):
            self.hide_key_phrases = True
        else:
            self.hide_key_phrases = False

        self.build_graph()

    def build_graph(self):
        """ Build the computation graph. """
        self._inputs = tf.placeholder(dtype=tf.int64, shape=[None, self.sent_len + 2], name='input_x')
        self._sentences = self._inputs[:,:-2]
        # Number of key phrases is always 2
        self._key_phrase_indices = self._inputs[:,-2:]
        self._labels = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_y')
        self._attention = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='attention')
        losses = []

        # lookup layer
        with tf.variable_scope('embedding') as scope:
            self._W_emb = _variable_on_cpu(name='embedding', shape=[self.vocab_size, self.word_emb_size],
                                           initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)) \
                if not self.use_gpu else _variable_on_gpu(name='embedding', shape=[self.vocab_size, self.word_emb_size],
                                           initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
            self._relative_distance_emb = _variable_on_cpu(name='relative_distance_embedding',
                                                           shape=[self.sent_len * 2 - 1, self.pos_emb_size],
                                                           initializer=tf.random_uniform_initializer(minval=-1.0,
                                                                                                     maxval=1.0)) \
                if not self.use_gpu else _variable_on_gpu(name='relative_distance_embedding',
                                                           shape=[self.sent_len * 2 - 1, self.pos_emb_size],
                                                           initializer=tf.random_uniform_initializer(minval=-1.0,
                                                                                                     maxval=1.0))

            # # sent_batch is of shape: (batch_size, sent_len, word_emb_size + 2 * pos_emb_size, 1), in order to use conv2d
            sent_batch = neural_util.embedding_lookup_in_sentence(self._sentences,self._key_phrase_indices,self._W_emb,
                                                                self._relative_distance_emb)
            sent_batch = tf.expand_dims(sent_batch, -1)
            # sent_batch = tf.nn.embedding_lookup(params=self._W_emb, ids=self._inputs)
            # sent_batch = tf.expand_dims(sent_batch, -1)

        # conv + pooling layer
        pool_tensors = []
        for k_size in range(self.min_window, self.max_window+1):
            with tf.variable_scope('conv-%d' % k_size) as scope:
                kernel, wd = _variable_with_weight_decay(
                    name='kernel-%d' % k_size,
                    shape=[k_size, self.word_emb_size + 2 * self.pos_emb_size, 1, self.num_kernel],
                    initializer=tf.truncated_normal_initializer(stddev=0.01),
                    wd=self.l2_reg)
                losses.append(wd)
                conv = tf.nn.conv2d(input=sent_batch, filter=kernel, strides=[1,1,1,1], padding='VALID')
                biases = _variable_on_cpu(name='bias-%d' % k_size,
                                          shape=[self.num_kernel],
                                          initializer=tf.constant_initializer(0.0)) \
                    if not self.use_gpu else _variable_on_gpu(name='bias-%d' % k_size,
                                                              shape=[self.num_kernel],
                                                              initializer=tf.constant_initializer(0.0))
                bias = tf.nn.bias_add(conv, biases)
                activation = tf.nn.relu(bias, name=scope.name)
                # shape of activation: [batch_size, conv_len, 1, num_kernel]
                conv_len = activation.get_shape()[1]
                pool = tf.nn.max_pool(activation, ksize=[1,conv_len,1,1], strides=[1,1,1,1], padding='VALID')
                # shape of pool: [batch_size, 1, 1, num_kernel]
                pool_tensors.append(pool)

        # Combine all pooled tensors
        num_filters = self.max_window - self.min_window + 1
        pool_size = num_filters * self.num_kernel
        pool_layer = tf.concat(num_filters, pool_tensors, name='pool')
        pool_flat = tf.reshape(pool_layer, [-1, pool_size])

        # drop out layer
        if self.is_train and self.dropout > 0:
            pool_dropout = tf.nn.dropout(pool_flat, 1 - self.dropout)
        else:
            pool_dropout = pool_flat

        # fully-connected layer
        with tf.variable_scope('output') as scope:
            W, wd = _variable_with_weight_decay('W', shape=[pool_size, self.num_classes],
                                                initializer=tf.truncated_normal_initializer(stddev=0.05),
                                                wd=self.l2_reg, use_gpu=self.use_gpu)
            losses.append(wd)
            biases = _variable_on_cpu('bias', shape=[self.num_classes],
                                      initializer=tf.constant_initializer(0.01)) \
                if not self.use_gpu else _variable_on_gpu('bias', shape=[self.num_classes],
                                      initializer=tf.constant_initializer(0.01))
            self.logits = tf.nn.bias_add(tf.matmul(pool_dropout, W), biases, name='logits')

        # loss
        with tf.variable_scope('loss') as scope:
            if self.multi_label:
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self._labels,
                                                                        name='cross_entropy_per_example')
            else:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, self._labels,
                                                                        name='cross_entropy_per_example')

            if self.is_train and self.multi_instance: # apply attention
                cross_entropy_loss = tf.reduce_sum(tf.mul(cross_entropy, self._attention),
                                                   name='cross_entropy_loss')
            else:
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
    def attention(self):
        return self._attention

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

    @property
    def W_emb(self):
        return self._W_emb

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    def assign_embedding(self, session, pretrained):
        session.run(tf.assign(self.W_emb, pretrained))
