from datetime import datetime
import hashlib
import time
import os
import tensorflow as tf
import numpy as np
import random

import cnn
import util

import train
import label
import preprocessing_util
import train_kp_pair_classifier
from active_learning_offline_util import save_additional_label
from infer_labels_util import infer_from_labeled

FLAGS = tf.app.flags.FLAGS

def _is_sentence_train(i):
    return i % 2 == 0

def get_hash_unlabeled_dataset(unlabeled_data):
    num = unlabeled_data.shape[0]
    ret = {}
    for i in range(num):
        ret[get_hash_unlabeled_item(unlabeled_data[i])]=i
    return ret

def get_hash_unlabeled_item(unlabeled_item):
    return hashlib.md5(unlabeled_item).hexdigest()

def find_corresponding_index_in_unlabeled(alternative_unlabeled_data, unlabeled_data, additional_label_index,):
    # For each additional label in the unlabeled_data, find the corresponding index in the alternative unlabeled data.
    ret = []
    alternative_unlabeled_data_hash_to_index = get_hash_unlabeled_dataset(alternative_unlabeled_data)
    for additional_label_i, index in enumerate(additional_label_index):
        additional_label_hash = get_hash_unlabeled_item(unlabeled_data[index])
        alternative_unlabeled_data_index = alternative_unlabeled_data_hash_to_index.get(additional_label_hash, None)
        if alternative_unlabeled_data_index is None:
            raise AssertionError("Cannot find the corresponding unlabeled data item for index %d. " %(index))
            continue
        ret.append(alternative_unlabeled_data_index)
    return ret

def draw_from_unused_unlabeled(unlabeled_data, used_unlabeled_data_indices, num_to_draw):
    # Return the drawn data and the new set of used_unlabeled_data_indices
    assert isinstance(used_unlabeled_data_indices, set)
    num_unlabeled_data = unlabeled_data.shape[0]
    unused_indices = set([i for i in range(num_unlabeled_data)]) - used_unlabeled_data_indices
    drawn_indices = random.sample(unused_indices, num_to_draw)
    drawn_data = unlabeled_data[drawn_indices]
    used_unlabeled_data_indices = used_unlabeled_data_indices.union(set(drawn_indices))
    return drawn_data, used_unlabeled_data_indices

def main(argv=None):
    # Flags are defined in train.py
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    latest_sentence_checkpoint_dir = None
    latest_pair_checkpoint_dir = None
    used_unlabeled_data_indices = set()


    for round_i in range(FLAGS.max_co_training_rounds):
        # load dataset
        # source_path = os.path.join(FLAGS.data_dir, 'ids.txt')
        # target_path = os.path.join(FLAGS.data_dir, 'target.txt')
        if round_i == 0:
            source_path = os.path.join(FLAGS.data_dir, 'test_cs_unlabeled_data_combined_inferred_train.txt')
            target_path = os.path.join(FLAGS.data_dir, 'test_cs_labels_combined_inferred_train.txt')
            unlabeled_data = util.read_data_unlabeled_part(source_path, target_path, FLAGS.sent_len, shuffle=False)

        else:
            if _is_sentence_train(round_i):
                source_path = os.path.join(latest_pair_checkpoint_dir, 'test_cs_unlabeled_data_combined_round_%d.txt' %(round_i))
                target_path = os.path.join(latest_pair_checkpoint_dir, 'test_cs_labels_combined_round_%d.txt' %(round_i))
            else:
                source_path = os.path.join(latest_sentence_checkpoint_dir, 'test_cs_unlabeled_data_combined_round_%d.txt' %(round_i))
                target_path = os.path.join(latest_sentence_checkpoint_dir, 'test_cs_labels_combined_round_%d.txt' %(round_i))
        # attention_path = None
        # if FLAGS.attention:
        #     if os.path.exists(os.path.join(FLAGS.data_dir, 'source.att')):
        #         attention_path = os.path.join(FLAGS.data_dir, 'source.att')
        #     else:
        #         raise ValueError("Attention file %s not found.", os.path.join(FLAGS.data_dir, 'source.att'))
        train_data, test_data = util.read_data(source_path, target_path, FLAGS.sent_len,
                                               attention_path=None, train_size=FLAGS.train_size,
                                               hide_key_phrases=FLAGS.hide_key_phrases)  # TODO: disable hide key phrases for training kp pair classifier.
        if _is_sentence_train(round_i):
            train.train(train_data, test_data)
        else:
            train_kp_pair_classifier.train(train_data,test_data)

        labeled_data, labeled_result = util.read_data_labeled_part(source_path, target_path, FLAGS.sent_len, shuffle=False)
        # For each round, we draw a fresh set of unlabeled data and label them using the trained classifier.
        current_unlabeled_data,used_unlabeled_data_indices  = draw_from_unused_unlabeled(unlabeled_data, used_unlabeled_data_indices, FLAGS.test_size_per_round)

        # Refresh the latest checkpoint.
        latest_checkpoint_dir = util.get_latest_checkpoint_dir(FLAGS.train_dir)
        restore_param = util.load_from_dump(os.path.join(latest_checkpoint_dir, 'flags.cPickle'))
        restore_param['train_dir'] = latest_checkpoint_dir
        if _is_sentence_train(round_i):
            x_input, actual_output = label.label(current_unlabeled_data, restore_param)
        else:
            x_input, actual_output = train_kp_pair_classifier.label(current_unlabeled_data, restore_param)


        actual_output_exp = np.exp(actual_output)
        actual_output_softmax = actual_output_exp / np.sum(actual_output_exp, axis=1, keepdims=True)
        actual_output_argmax = np.argmax(actual_output_softmax, axis=1)
        # If we do not want "Neither" relation, then calculate max on only the first 2 dimensions.
        # actual_output_softmax_sorted = np.argsort(-np.max(actual_output_softmax[..., :2], axis=1)).tolist()
        actual_output_softmax_sorted = np.argsort(-np.max(actual_output_softmax, axis=1)).tolist()

        sentence_indices_input = x_input[:, :-2]
        vocab_path = os.path.join(restore_param['data_dir'], 'test_cs_vocab_combined')
        _, rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)
        sentence_input = preprocessing_util.indices_to_sentences(sentence_indices_input, rev_vocab)

        kp_indices_input = x_input[:, -2:]

        print('Type\tSentence\t\tProbability [A is-a B, B is-a A, Neither]')
        # for sentence_i, sentence in enumerate(sentence_input):
        additional_label_index = []
        additional_label_result = []
        min_threshold = min(FLAGS.co_training_has_relation_prob_upper_threshold,
                            FLAGS.co_training_no_relation_prob_upper_threshold)
        for sentence_i in actual_output_softmax_sorted:
            # This is the current max probability
            current_softmax = actual_output_softmax[sentence_i,actual_output_argmax[sentence_i]]
            # Stop when this probability drops below 0.8
            if current_softmax < min_threshold:
                break

            sentence = sentence_input[sentence_i]
            # Label the key phrases of interest in the current sentence with *.
            sentence[kp_indices_input[sentence_i, 1]] += '*'
            sentence[kp_indices_input[sentence_i, 0]] += '*'
            if actual_output_argmax[sentence_i] == 2:
                current_type = 'Neither'
                if current_softmax < FLAGS.co_training_no_relation_prob_upper_threshold:
                    continue
            if actual_output_argmax[sentence_i] == 0:
                current_type = 'A is-a B'
                if current_softmax < FLAGS.co_training_has_relation_prob_upper_threshold:
                    continue
            elif actual_output_argmax[sentence_i] == 1:
                current_type = 'B is-a A'
                if current_softmax < FLAGS.co_training_has_relation_prob_upper_threshold:
                    continue

            print('%s\t%s\t%f\t\t%s\t'
                  % (current_type, ' '.join(sentence), current_softmax, str(actual_output_softmax[sentence_i])))

            additional_label_index.append(sentence_i)
            current_additional_label_result = np.zeros((3,))
            current_additional_label_result[actual_output_argmax[sentence_i]] = 1
            additional_label_result.append(current_additional_label_result)

        # TODO: alternate between the two sources.
        print("Number of additional data points added through co-training: %d out of %d unlabeled instances."
              %(len(additional_label_index), len(actual_output_softmax_sorted)))

        # Now I should read the dataset from the other classifier - the one different from what was just trained - and
        # write the additional result into that dataset.
        if round_i <= 2:
            alternative_source_path = os.path.join(FLAGS.data_dir, 'test_cs_unlabeled_data_combined_inferred_train.txt')
            alternative_target_path = os.path.join(FLAGS.data_dir, 'test_cs_labels_combined_inferred_train.txt')
        else:
            if _is_sentence_train(round_i):
                alternative_source_path = os.path.join(latest_sentence_checkpoint_dir, 'test_cs_unlabeled_data_combined_round_%d.txt' %(round_i - 1))
                alternative_target_path = os.path.join(latest_sentence_checkpoint_dir, 'test_cs_labels_combined_round_%d.txt' %(round_i - 1))
            else:
                alternative_source_path = os.path.join(latest_pair_checkpoint_dir, 'test_cs_unlabeled_data_combined_round_%d.txt' %(round_i - 1))
                alternative_target_path = os.path.join(latest_pair_checkpoint_dir, 'test_cs_labels_combined_round_%d.txt' %(round_i - 1))
        alternative_labeled_data, alternative_labeled_result = util.read_data_labeled_part(alternative_source_path, alternative_target_path, FLAGS.sent_len,
                                                                   shuffle=False)
        alternative_unlabeled_data = util.read_data_unlabeled_part(alternative_source_path, alternative_target_path, FLAGS.sent_len, shuffle=False)
        save_source_path = os.path.join(latest_checkpoint_dir, 'test_cs_unlabeled_data_combined_round_%d.txt' % (round_i + 1))
        save_target_path = os.path.join(latest_checkpoint_dir, 'test_cs_labels_combined_round_%d.txt' % (round_i + 1))
        # save_additional_label(current_unlabeled_data, additional_label_index, additional_label_result, labeled_data, labeled_result,
        #                       save_source_path, save_target_path)

        # But the additional index and the unlabeled data might not match each other!
        # TODO: how does co-training actually work? Should the test data be the unlabeled data of the other classifier?
        # Or should it be the unlabeled data of the current classifier? It make sense to use the current classifier.
        # because otherwise the test data may contain training data. So I need to combine the two.
        alternative_additional_label_index = find_corresponding_index_in_unlabeled(alternative_unlabeled_data, current_unlabeled_data, additional_label_index, )
        save_additional_label(alternative_unlabeled_data, alternative_additional_label_index, additional_label_result, alternative_labeled_data,
                              alternative_labeled_result, save_source_path, save_target_path)

        # I also need to get rid of those inferred instances from the whole bag of unlabeled dataset that we're drawing
        # from at each round.
        before_inference_unlabeled_data = util.read_data_unlabeled_part(save_source_path, save_target_path, FLAGS.sent_len, shuffle=False)
        inferred_additional_label_index, inferred_additional_label_result = infer_from_labeled(save_source_path,
                                                                                               save_target_path,
                                                                                               FLAGS.sent_len, vocab_path,
                                                                                               do_save=True,
                                                                                               save_source_path=save_source_path,
                                                                                               save_target_path=save_target_path)

        inferred_additional_label_index_in_original = \
            find_corresponding_index_in_unlabeled(unlabeled_data, before_inference_unlabeled_data,
                                                  inferred_additional_label_index, )
        used_unlabeled_data_indices = used_unlabeled_data_indices.union(set(inferred_additional_label_index_in_original))




        if _is_sentence_train(round_i):
            latest_sentence_checkpoint_dir = latest_checkpoint_dir
        else:
            latest_pair_checkpoint_dir = latest_checkpoint_dir


if __name__ == '__main__':
    cnn.define_flags()
    train.define_flags()
    tf.app.flags.DEFINE_float('co_training_has_relation_prob_upper_threshold', 0.9,
                              "The upper threshold for the probability of belonging to a class. If the probability of "
                              "an unlabeled instance is above this threshold, it will be used as extra training data "
                              "for co training.")
    tf.app.flags.DEFINE_float('co_training_no_relation_prob_upper_threshold', 0.9999,
                              "The threshold for the probability of belonging to no relation `Neither`. If the "
                              "probability of an unlabeled instance is above this threshold, it will be used as extra "
                              "training data for co training.")
    tf.app.flags.DEFINE_integer('max_co_training_rounds', 10,
                                "The maximum number of rounds for co training.")
    tf.app.flags.DEFINE_integer('test_size_per_round', 1000,
                                "The number of instance drawn from the unlabeled dataset per round for co training.")

    tf.app.run()