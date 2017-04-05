import os

import numpy as np
import tensorflow as tf

import preprocessing_util
from util import read_data_labeled_part, read_data_unlabeled_part, load_from_dump
from active_learning_offline_util import save_additional_label


def get_all_kp_pair_from_labeled(labeled_data, labeled_result, rev_vocab):
    """

    :param labeled_data: np array representing labeled sentences.
    :param labeled_result: np array representing the labels.
    :return: a dictionary with key = (key phrase 1, key phrase 2) and value = the label marked for that key phrase. If
    there are multiple labels and they are different, throw assertion error?
    """
    sentence_indices_input = labeled_data[:,:-2]
    kp_indices_input = labeled_data[:,-2:]
    key_phrases = {}
    for i in range(labeled_data.shape[0]):
        current_kp = tuple(sentence_indices_input[i,kp_indices_input[i,:]])
        current_label = labeled_result[i]
        if current_kp in key_phrases:
            if np.any(key_phrases[current_kp] != current_label):
                raise AssertionError("The key phrase pair %s:%s does not have a consistent label. "
                                     "Previously encountered label was %s and it is currently %s."
                                     %(rev_vocab[current_kp[0]], rev_vocab[current_kp[1]],
                                       str(key_phrases[current_kp]), str(current_label)))
        else:
            key_phrases[current_kp] = current_label
    return key_phrases


def infer_from_labeled(source_path, target_path, sentence_length, vocab_path, save_source_path = None, save_target_path = None, save_unlabeled_kp_path=None):
    if save_source_path is None:
        save_source_path = os.path.splitext(source_path)[0] + "_inferred.txt"
    if save_target_path is None:
        save_target_path = os.path.splitext(target_path)[0] + "_inferred.txt"
    if save_unlabeled_kp_path is None:
        save_unlabeled_kp_path = os.path.join(os.path.dirname(source_path), "unlabeled_kp.npy")


    labeled_data, labeled_result = read_data_labeled_part(source_path, target_path, sentence_length, shuffle=False)
    labeled_data = np.array(labeled_data,dtype=np.int)
    labeled_result = np.array(labeled_result,dtype=np.int)
    # sentence_indices_input = labeled_data[:,:-2]
    # kp_indices_input = labeled_data[:,-2:]

    unlabeled_data = read_data_unlabeled_part(source_path, target_path, sentence_length, shuffle=False)
    unlabeled_data = np.array(unlabeled_data,dtype=np.int)
    sentence_indices_unlabeled_data = unlabeled_data[:,:-2]
    kp_indices_unlabeled_data = unlabeled_data[:,-2:]


    vocab,rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)

    num_labeled_data = labeled_data.shape[0]
    num_unlabeled_data = unlabeled_data.shape[0]

    kp_label_dict = get_all_kp_pair_from_labeled(labeled_data, labeled_result, rev_vocab)

    # Now go through all unlabeled data, check whether their key phrase pairs appeared before, and if so, apply the
    # labeled result on the unlabeled pairs, because we assume that if one key phrase is labeled once, its label will
    # not change regardless of what sentence it is in.
    additional_label_index = []
    additional_label_result = []
    unlabeled_kp = set()
    for unlabeled_i in range(num_unlabeled_data):

        current_kp = tuple(sentence_indices_unlabeled_data[unlabeled_i,kp_indices_unlabeled_data[unlabeled_i,:]])
        current_label = kp_label_dict.get(current_kp, None)
        if current_label is not None:
            additional_label_index.append(unlabeled_i)
            additional_label_result.append(current_label)
        else:
            if current_kp not in unlabeled_kp:
                unlabeled_kp.add(current_kp)

    print('Labeled %d additional sentences out of %d unlabeled sentences. There are %d unlabeled key phrase pairs left.'
          % (len(additional_label_result), num_unlabeled_data, len(unlabeled_kp)))
    save_additional_label(unlabeled_data, additional_label_index, additional_label_result, labeled_data, labeled_result,
                          save_source_path, save_target_path)
    unlabeled_kp = np.array(list(unlabeled_kp), dtype=np.int32)
    np.save(save_unlabeled_kp_path, unlabeled_kp)

def save_all_unlabeled_kp_pair(source_path, target_path, sentence_length, vocab_path, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(source_path), "unlabeled_kp.npy")

    labeled_data, labeled_result = read_data_labeled_part(source_path, target_path, sentence_length, shuffle=False)
    labeled_data = np.array(labeled_data, dtype=np.int)
    labeled_result = np.array(labeled_result, dtype=np.int)
    # sentence_indices_input = labeled_data[:,:-2]
    # kp_indices_input = labeled_data[:,-2:]

    unlabeled_data = read_data_unlabeled_part(source_path, target_path, sentence_length, shuffle=False)
    unlabeled_data = np.array(unlabeled_data, dtype=np.int)
    sentence_indices_unlabeled_data = unlabeled_data[:, :-2]
    kp_indices_unlabeled_data = unlabeled_data[:, -2:]

    vocab, rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)

    num_labeled_data = labeled_data.shape[0]
    num_unlabeled_data = unlabeled_data.shape[0]

    kp_label_dict = get_all_kp_pair_from_labeled(labeled_data, labeled_result, rev_vocab)

    # Now go through all unlabeled data, check whether their key phrase pairs appeared before, and if so, apply the
    # labeled result on the unlabeled pairs, because we assume that if one key phrase is labeled once, its label will
    # not change regardless of what sentence it is in.
    unlabeled_kp = set()
    for unlabeled_i in range(num_unlabeled_data):

        current_kp = tuple(sentence_indices_unlabeled_data[unlabeled_i, kp_indices_unlabeled_data[unlabeled_i, :]])
        current_label = kp_label_dict.get(current_kp, None)
        if current_label is None and current_kp not in unlabeled_kp:
            unlabeled_kp.add(current_kp)

    print('There are %d unlabeled key phrases out of %d unlabeled sentences.'
          % (len(unlabeled_kp), num_unlabeled_data))
    unlabeled_kp = np.array(list(unlabeled_kp), dtype=np.int32)
    np.save(save_path, unlabeled_kp)


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS

    this_dir = os.path.abspath(os.path.dirname(__file__))

    # eval parameters
    tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'checkpoints'), 'Directory of the checkpoint files')

    restore_param = load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir
    restore_param['data_dir'] = "inferred_dataset"
    source_path = os.path.join(restore_param['data_dir'], 'test_cs_unlabeled_data_combined.txt')
    target_path = os.path.join(restore_param['data_dir'], 'test_cs_labels_combined.txt')
    vocab_path = os.path.join(restore_param['data_dir'], 'test_cs_vocab_combined')
    infer_from_labeled(source_path,target_path,restore_param['sent_len'],vocab_path)