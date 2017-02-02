# -*- coding: utf-8 -*-

##########################################################
#
# Helper functions to load data
# Note this file is mostly took from https://github.com/may-/cnn-re-tf/blob/master/util.py
#
###########################################################

import os
import re
from codecs import open as codecs_open
import cPickle as pickle
import numpy as np

import preprocessing_util


# Special vocabulary symbols.
PAD_TOKEN = '<pad>' # pad symbol
UNK_TOKEN = '<unk>' # unknown word
BOS_TOKEN = '<bos>' # begin-of-sentence symbol
EOS_TOKEN = '<eos>' # end-of-sentence symbol
NUM_TOKEN = '<num>' # numbers

# we always put them at the start.
_START_VOCAB = [PAD_TOKEN, UNK_TOKEN]
PAD_ID = 0
UNK_ID = 1

# Regular expressions used to tokenize.
_DIGIT_RE = re.compile(br"^\d+$")

# DEFAULT_RANDOM_SEED
DEFAULT_RANDOM_SEED = 1234

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def basic_tokenizer(sequence, bos=True, eos=True):
    sequence = re.sub(r'\s{2}', ' ' + EOS_TOKEN + ' ' + BOS_TOKEN + ' ', sequence)
    if bos:
        sequence = BOS_TOKEN + ' ' + sequence.strip()
    if eos:
        sequence = sequence.strip() + ' ' + EOS_TOKEN
    return sequence.lower().split()


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=40000, tokenizer=None, bos=True, eos=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if not os.path.exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with codecs_open(data_path, "rb", encoding="utf-8") as f:
            for line in f.readlines():
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line, bos, eos)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, NUM_TOKEN, w)
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                print("  %d words found. Truncate to %d." % (len(vocab_list), max_vocabulary_size))
                vocab_list = vocab_list[:max_vocabulary_size]
            with codecs_open(vocabulary_path, "wb", encoding="utf-8") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")



def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with codecs_open(vocabulary_path, "rb", encoding="utf-8") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, bos=True, eos=True):
    """Convert a string to list of integers representing token-ids.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    words = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence, bos, eos)
    return [vocabulary.get(re.sub(_DIGIT_RE, NUM_TOKEN, w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None, bos=True, eos=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if not os.path.exists(target_path):
        print("Vectorizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with codecs_open(data_path, "rb", encoding="utf-8") as data_file:
            with codecs_open(target_path, "wb", encoding="utf-8") as tokens_file:
                for line in data_file:
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer, bos, eos)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def shuffle_split(X, y, a=None, train_size=10000, shuffle=True, random_seed = None):
    """Shuffle and split data into train and test subset"""
    _X = np.array(X)
    _y = np.array(y)
    assert _X.shape[0] == _y.shape[0]

    _a = [None] * _y.shape[0]
    if a is not None and len(a) == len(y):
        _a = np.array(a)
        # compute softmax
        _a = np.reshape(np.exp(_a) / np.sum(np.exp(_a)), (_y.shape[0], 1))
        assert _a.shape[0] == _y.shape[0]

    print "Splitting data...",
    # split train-test
    data = np.array(zip(_X, _y, _a))
    data_size = _y.shape[0]
    if train_size > data_size:
        train_size = int(data_size * 0.9)
    if shuffle:
        np.random.seed(random_seed)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    print "\t%d for train, %d for test" % (train_size, data_size - train_size)
    return shuffled_data[:train_size], shuffled_data[train_size:]

def read_labels_file(labels_path):
    with codecs_open(labels_path, mode="r", encoding="utf-8-sig") as target_file:
        ret = []
        line = target_file.readline()
        while line:
            labels = line.strip('\n').split(' ')
            if len(labels) == 3:
                ret.append([int(labels[0]),int(labels[1]),int(labels[2])])
            elif len(labels) == 0:
                pass
            else:
                raise AssertionError('Labels have the wrong format.')
            line = target_file.readline()
        return ret

def save_labels_file(labels_list,labels_save_path):
    with codecs_open(labels_save_path, mode="w", encoding="utf-8-sig") as target_file:

        if not isinstance(labels_list, list):
            assert isinstance(labels_list, np.ndarray)
            labels_iter = labels_list.astype(dtype=np.uint8).tolist()
        else:
            labels_iter = labels_list.astype(dtype=np.uint8)

        for labels in labels_iter:
            labels_str = map(str,labels)
            target_file.write(' '.join(labels_str) + '\n')

def read_data(source_path, target_path, sent_len, attention_path=None, train_size=10000, shuffle=True, random_seed=DEFAULT_RANDOM_SEED):
    """Read source(x), target(y) and attention if given.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py
    """
    _X = []
    _y = []
    # with codecs_open(source_path, mode="r", encoding="utf-8") as source_file:
    #     with codecs_open(target_path, mode="r", encoding="utf-8") as target_file:
    with codecs_open(source_path, mode="r") as source_file:
        with codecs_open(target_path, mode="r", encoding="utf-8-sig") as target_file:
            source, target = source_file.readline(), target_file.readline()
            #counter = 0
            print "Loading data...",
            while source and target:
                #counter += 1
                #if counter % 1000 == 0:
                #    print("  reading data line %d" % counter)
                #    sys.stdout.flush()
                source_ids = [np.int64(x.strip()) for x in source.split()]
                # The data should already be padded so it doesn;t need to do padding here.
                # if sent_len > len(source_ids):
                #     source_ids += [PAD_ID] * (sent_len - len(source_ids))
                if len(source_ids) != sent_len + 2:  # 2 represent the 2 indices for the key phrases.
                    raise AssertionError("At line %d the length of input source %d is not the same as the length of "
                                         "sentence plus 2, which is %d." %(len(_X),len(source_ids),sent_len+2))

                #target = target.split('\t')[0].strip()
                target_ids = [np.float32(y.strip()) for y in target.split()]

                _X.append(source_ids)
                _y.append(target_ids)
                source, target = source_file.readline(), target_file.readline()

    assert len(_X) == len(_y)
    print "\t%d examples found." % len(_y)

    _a = None
    if attention_path is not None:
        with codecs_open(attention_path, mode="r", encoding="utf-8") as att_file:
            _a = [np.float32(att.strip()) for att in att_file.readlines()]
            assert len(_a) == len(_y)

    return shuffle_split(_X, _y, a=_a, train_size=train_size, shuffle=shuffle, random_seed=random_seed)


def read_data_unlabeled_part(source_path, target_path, sent_len, shuffle=True, shuffle_seed = None):
    """Read source(x), target(y) and attention if given.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py
    """
    _X = []
    _y = []
    # with codecs_open(source_path, mode="r", encoding="utf-8") as source_file:
    #     with codecs_open(target_path, mode="r", encoding="utf-8") as target_file:
    with codecs_open(source_path, mode="r") as source_file:
        with codecs_open(target_path, mode="r", encoding="utf-8-sig") as target_file:
            source, target = source_file.readline(), target_file.readline()
            #counter = 0
            print "Loading data...",
            while source and target:
                source, target = source_file.readline(), target_file.readline()
                continue

            while source:
                #counter += 1
                #if counter % 1000 == 0:
                #    print("  reading data line %d" % counter)
                #    sys.stdout.flush()
                source_ids = [np.int64(x.strip()) for x in source.split()]
                # The data should already be padded so it doesn;t need to do padding here.
                # if sent_len > len(source_ids):
                #     source_ids += [PAD_ID] * (sent_len - len(source_ids))
                assert len(source_ids) == sent_len + 2  # 2 represent the 2 indices for the key phrases.


                _X.append(source_ids)
                source = source_file.readline()

    data_size = len(_X)
    print "\t%d examples found." % data_size
    data = np.array(_X)
    if shuffle:
        np.random.seed(seed=shuffle_seed)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
    return data



def read_data_labeled_part(source_path, target_path, sent_len, shuffle=True, shuffle_seed = None):
    """Read source(x), target(y) and attention if given.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py
    """
    _X = []
    _y = []
    # with codecs_open(source_path, mode="r", encoding="utf-8") as source_file:
    #     with codecs_open(target_path, mode="r", encoding="utf-8") as target_file:
    with codecs_open(source_path, mode="r") as source_file:
        with codecs_open(target_path, mode="r", encoding="utf-8-sig") as target_file:
            source, target = source_file.readline(), target_file.readline()
            #counter = 0
            print "Loading data...",
            while source and target:
                # counter += 1
                # if counter % 1000 == 0:
                #    print("  reading data line %d" % counter)
                #    sys.stdout.flush()
                source_ids = [np.int64(x.strip()) for x in source.split()]
                # The data should already be padded so it doesn;t need to do padding here.
                # if sent_len > len(source_ids):
                #     source_ids += [PAD_ID] * (sent_len - len(source_ids))
                assert len(source_ids) == sent_len + 2  # 2 represent the 2 indices for the key phrases.

                # target = target.split('\t')[0].strip()
                target_ids = [np.float32(y.strip()) for y in target.split()]

                _X.append(source_ids)
                _y.append(target_ids)
                source, target = source_file.readline(), target_file.readline()

    data_size = len(_X)
    assert len(_X) == len(_y)
    print "\t%d examples found." % data_size


    if shuffle:
        np.random.seed(seed=shuffle_seed)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        _X = _X[shuffle_indices]
        _y = _y[shuffle_indices]
    return _X, _y

def shuffle_split_contextwise(X, y, a=None, train_size=10000, shuffle=True, random_seed = None):
    """Shuffle and split data into train and test subset"""

    _left = np.array(X['left'])
    _middle = np.array(X['middle'])
    _right = np.array(X['right'])
    _y = np.array(y)

    _a = [None] * _y.shape[0]
    if a is not None and len(a) == len(y):
        _a = np.array(a)
        # compute softmax
        _a = np.reshape(np.exp(_a) / np.sum(np.exp(_a)), (_y.shape[0], 1))
        assert _a.shape[0] == _y.shape[0]

    print "Splitting data...",
    # split train-test
    data = np.array(zip(_left, _middle, _right, _y, _a))
    data_size = _y.shape[0]
    if train_size > data_size:
        train_size = int(data_size * 0.9)
    if shuffle:
        np.random.seed(random_seed)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    print "\t%d for train, %d for test" % (train_size, data_size - train_size)
    return shuffled_data[:train_size], shuffled_data[train_size:]


def read_data_contextwise(source_path, target_path, sent_len, attention_path=None, train_size=10000, shuffle=True):
    """Read source file and pad the sequence to sent_len,
       combine them with target (and attention if given).

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py
    """
    print "Loading data...",
    _X = {'left': [], 'middle': [], 'right': []}
    for context in _X.keys():
        path = '%s.%s' % (source_path, context)
        with codecs_open(path, mode="r", encoding="utf-8") as source_file:
            for source in source_file.readlines():
                source_ids = [np.int64(x.strip()) for x in source.split()]
                if sent_len > len(source_ids):
                    source_ids += [PAD_ID] * (sent_len - len(source_ids))
                assert len(source_ids) == sent_len
                _X[context].append(source_ids)
    assert len(_X['left']) == len(_X['middle'])
    assert len(_X['right']) == len(_X['middle'])

    _y = []
    with codecs_open(target_path, mode="r", encoding="utf-8") as target_file:
        for target in target_file.readlines():
            target_ids = [np.float32(y.strip()) for y in target.split()]
            _y.append(target_ids)
    assert len(_X['left']) == len(_y)
    print "\t%d examples found." % len(_y)

    _a = None
    if attention_path is not None:
        with codecs_open(attention_path, mode="r", encoding="utf-8") as att_file:
            _a = [np.float32(att.strip()) for att in att_file.readlines()]
            assert len(_a) == len(_y)

    return shuffle_split_contextwise(_X, _y, a=_a, train_size=train_size, shuffle=shuffle)



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Generates a batch iterator.

    Original taken from
    https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(np.ceil(float(data_size)/batch_size))
    for epoch in range(num_epochs):
        # Shuffle data at each epoch
        if shuffle:
            #np.random.seed(RANDOM_SEED)
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



def dump_to_file(filename, obj):
    with open(filename, 'wb') as outfile:
        pickle.dump(obj, file=outfile)
    return

def load_from_dump(filename):
    with open(filename, 'rb') as infile:
        obj = pickle.load(infile)
    return obj

def _load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec

    Original taken from
    https://github.com/yuhaozhang/sentence-convnet/blob/master/text_input.py
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return (word_vecs, layer1_size)

def _add_random_vec(word_vecs, vocab, emb_size=300):
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,emb_size)
    return word_vecs

def _add_phrase_vec_and_random_vec(word_vecs, vocab, emb_size=300):
    """
    This function returns the word vectors with the vectors for phrases in vocab added. The phrase vectors are
    made from the element wise sum of the individual words. If individual word does not exist, a random vector is
    created for that word.
    :param word_vecs:
    :param vocab:
    :param emb_size:
    :return:
    """
    for word in vocab:
        if word not in word_vecs:
            if '_' in word:
                components = word.split('_')
                word_vec = np.zeros((emb_size))
                for component in components:
                    if component not in word_vecs:
                        word_vecs[component] = np.random.uniform(-0.25, 0.25, emb_size)
                    component_vec = word_vecs[component]
                    word_vec = np.add(word_vec,component_vec)
                word_vecs[word] = word_vec
            else:
                word_vecs[word] = np.random.uniform(-0.25, 0.25, emb_size)
    return word_vecs

def prepare_pretrained_embedding(fname, word2id):
    print 'Reading pretrained word vectors from file ...'
    word_vecs, emb_size = _load_bin_vec(fname, word2id)
    # Replacing the add_random_vec function with a new one that supports phrase vectors.
    # word_vecs = _add_random_vec(word_vecs, word2id, emb_size)
    word_vecs = _add_phrase_vec_and_random_vec(word_vecs, word2id, emb_size)
    embedding = np.zeros([len(word2id), emb_size])
    for w,idx in word2id.iteritems():
        embedding[idx,:] = word_vecs[w]
    print 'Generated embeddings with shape ' + str(embedding.shape)
    return embedding




def offset(array, pre, post):
    ret = np.array(array)
    ret = np.insert(ret, 0, pre)
    ret = np.append(ret, post)
    return ret

def calc_auc_pr(precision, recall):
    assert len(precision) == len(recall)
    return np.trapz(offset(precision, 1, 0), x=offset(recall, 0, 1), dx=5)

def prepare_ids(data_dir, vocab_path):
    for context in ['left', 'middle', 'right', 'txt']:
        data_path = os.path.join(data_dir, 'mlmi', 'source.%s' % context)
        target_path = os.path.join(data_dir, 'mlmi', 'ids.%s' % context)
        if context == 'left':
            bos, eos = True, False
        elif context == 'middle':
            bos, eos = False, False
        elif context == 'right':
            bos, eos = False, True
        else:
            bos, eos = True, True
        data_to_token_ids(data_path, target_path, vocab_path, bos=bos, eos=eos)


def get_valid_user_input(user_input,lower_threshold, upper_threshold):
    valid_user_input = False
    while not valid_user_input:
        try:
            user_input = int(user_input)
            if user_input > upper_threshold or user_input < lower_threshold:
                valid_user_input = False
            else:
                valid_user_input = True
        except:
            valid_user_input = False
        if not valid_user_input:
            user_input = raw_input('Please enter a valid integer %d ~ %d!' %(lower_threshold, upper_threshold))
    return user_input

def modify_labeled_data(source_path, target_path, kp_pair, sentence_length, vocab_path):

    labeled_data, labeled_result = read_data_labeled_part(source_path, target_path, sentence_length, shuffle=False)
    labeled_data = np.array(labeled_data,dtype=np.int)
    labeled_result = np.array(labeled_result,dtype=np.int)
    sentence_indices_input = labeled_data[:,:-2]
    kp_indices_input = labeled_data[:,-2:]



    vocab,rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)
    kp_vocab_i_pair = (vocab[kp_pair[0]],vocab[kp_pair[1]])

    num_labeled_data = labeled_data.shape[0]

    matching_indices = []

    for labeled_i in range(num_labeled_data):
        current_kp = sentence_indices_input[labeled_i,kp_indices_input[labeled_i]]
        if current_kp[0] == kp_vocab_i_pair[0] and current_kp[1] == kp_vocab_i_pair[1]:
            matching_indices.append(labeled_i)

    print('There are %d sentences with key phrase pair %s - %s' %(len(matching_indices), kp_pair[0], kp_pair[1]))


    matching_sentences = preprocessing_util.indices_to_sentences(sentence_indices_input[matching_indices], rev_vocab)
    for i, matching_index in enumerate(matching_indices):
        sentence = matching_sentences[i]
        sentence[kp_indices_input[matching_index, 1]] += '*'
        sentence[kp_indices_input[matching_index, 0]] += '*'

        print('%s\t\t%s\t'
              % (' '.join(sentence), str(labeled_result[matching_index])))

    user_input = raw_input('Would you like to change their labels?'
                           'Please enter 1, 2, or 3. Enter 4 to terminate.\n'
                           '1. A is-a B\n2. B is-a A\n3. Neither.')
    user_input = get_valid_user_input(user_input, 1, 4)

    if user_input == 4:
        return
    else:
        user_label_result = np.array([0, 0, 0])
        user_label_result[user_input - 1] = 1

        labeled_result[matching_indices] = user_label_result

        # Should I also get all kp pairs in the unlabeled part and label them? probably not since some words have more
        # than one meaning
        unlabeled_data = read_data_unlabeled_part(source_path, target_path, sentence_length)

        data_x = np.concatenate((labeled_data, unlabeled_data), axis=0)
        data_y = labeled_result

        np.savetxt(source_path, data_x, delimiter=' ', fmt='%d')
        save_labels_file(data_y, target_path)

def get_latest_checkpoint_dir(parent_dir):
    immediate_subdir = [name for name in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, name))]
    assert len(immediate_subdir) > 0
    immediate_subdir = sorted(immediate_subdir)
    return os.path.join(parent_dir, immediate_subdir[-1])  # Last one is the latest.



def main():
    data_dir = os.path.join(THIS_DIR, 'data')

    # multi-label multi-instance (MLMI-CNN) dataset
    vocab_path = os.path.join(data_dir, 'mlmi', 'vocab.txt')
    data_path = os.path.join(data_dir, 'mlmi', 'source.txt')
    max_vocab_size = 36500
    create_vocabulary(vocab_path, data_path, max_vocab_size)
    prepare_ids(data_dir, vocab_path)

    # pretrained embeddings
    embedding_path = os.path.join(THIS_DIR, 'word2vec', 'GoogleNews-vectors-negative300.bin')
    if os.path.exists(embedding_path):
        word2id, _ = initialize_vocabulary(vocab_path)
        embedding = prepare_pretrained_embedding(embedding_path, word2id)
        np.save(os.path.join(data_dir, 'mlmi', 'emb.npy'), embedding)
    else:
        print "Pretrained embeddings file %s not found." % embedding_path

    # single-label single-instance (ER-CNN) dataset
    vocab_er = os.path.join(data_dir, 'er', 'vocab.txt')
    data_er = os.path.join(data_dir, 'er', 'source.txt')
    target_er = os.path.join(data_dir, 'er', 'ids.txt')
    max_vocab_size = 11500
    tokenizer = lambda x: x.split()
    create_vocabulary(vocab_er, data_er, max_vocab_size, tokenizer=tokenizer)
    data_to_token_ids(data_er, target_er, vocab_er, tokenizer=tokenizer)


if __name__ == '__main__':
    main()
