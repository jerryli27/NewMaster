"""
This file contains functions used for active learning.
"""


from datetime import datetime
import os
import tensorflow as tf
import numpy as np
import pandas as pd


import util
import preprocessing_util
from label import label

NUM_TO_READ = 1000

def save_additional_label(unlabeled_data, additional_label_index, additional_label_result, labeled_data, labels, source_path,
                          target_path):
    if isinstance(additional_label_index,int):
        additional_label_index = [additional_label_index]
    if len(additional_label_index) == 0:
        data_x = np.concatenate((labeled_data, unlabeled_data),axis=0)
        data_y  = np.array(labels)

    else:
        additional_label_index_set = set(additional_label_index)
        num_unlabeled = unlabeled_data.shape[0]
        indices_not_included = [i for i in range(num_unlabeled) if i not in additional_label_index_set]
        data_x = np.concatenate((labeled_data, unlabeled_data[additional_label_index,...], unlabeled_data[indices_not_included,...]),axis=0)
        data_y  = np.concatenate((labels, additional_label_result), axis=0)

    np.savetxt(source_path, data_x, delimiter=' ', fmt='%d')
    util.save_labels_file(data_y,target_path)

def main(argv=None):
    restore_param = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir

    source_path = os.path.join(restore_param['data_dir'], 'test_cs_unlabeled_data_combined.txt')
    target_path = os.path.join(restore_param['data_dir'], 'test_cs_labels_combined.txt')
    vocab_path = os.path.join(restore_param['data_dir'], 'test_cs_vocab_combined')
    unlabeled_data = util.read_data_unlabeled_part(source_path, target_path, restore_param['sent_len'])
    data_size = unlabeled_data.shape[0]

    # # Now hard code to take the first 1000
    # data_first_1000 = unlabeled_data

    x_input, actual_output = label(unlabeled_data, restore_param)

    actual_output_exp = np.exp(actual_output)
    actual_output_softmax = actual_output_exp / np.sum(actual_output_exp, axis=1, keepdims=True)
    actual_output_argmax = np.argmax(actual_output_softmax,axis=1)
    # Entropy = -sum(p * log p) so this is actually the negative of entropy. For sorting purpose I took out the neg.
    actual_output_entropy = np.sum(np.multiply(actual_output_softmax, np.log(actual_output_softmax)), axis=1)

    # The following are ways to rank what question should be asked first.
    # The first one uses entropy, but there might be some implementation errors.
    actual_output_entropy_argsort = np.argsort(actual_output_entropy, axis=0) # This doesn:t seem to give me the most uncertain ones??? in theory it does. or maybe it's just the model is too sure of everything.
    # The second one uses the softmax probability and only ask the one with highest probability in the first two
    # classes.
    # actual_output_entropy_argsort = np.argsort(-np.max(actual_output_softmax[...,:2], axis=1))

    sentence_indices_input = x_input[:,:-2]
    _,rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)
    sentence_input = preprocessing_util.indices_to_sentences(sentence_indices_input,rev_vocab)

    kp_indices_input = x_input[:,-2:]
    #
    # print('Sentence\t\tPredicted Score (A is-a B, B is-a A, Neither)\t')
    # for sentence_i, sentence in enumerate(sentence_input):
    #     # Label the key phrases of interest in the current sentence with *.
    #     sentence[kp_indices_input[sentence_i,1]] += '*'
    #     sentence[kp_indices_input[sentence_i,0]] += '*'
    #     if actual_output_argmax[sentence_i] == 2:
    #         # current_type = 'Neither'
    #         continue
    #     if actual_output_argmax[sentence_i] == 0:
    #         current_type = 'A is-a B'
    #     elif actual_output_argmax[sentence_i] == 1:
    #         current_type = 'B is-a A'
    #
    #     print('%s\t%s\t\t%s\t'
    #           % (current_type, ' '.join(sentence), str(actual_output_softmax[sentence_i])))
    user_input = -1
    num_user_labeled = 0
    user_label_results = []
    while user_input != 4 and num_user_labeled < data_size:
        sentence_i = actual_output_entropy_argsort[num_user_labeled]
        sentence = sentence_input[sentence_i]
        print('Key phrase pair\tSentence\t\tPredicted Score (A is-a B, B is-a A, Neither)\t')

        current_key_phrase_pair = sentence[kp_indices_input[sentence_i,0]] + ' ' + sentence[kp_indices_input[sentence_i,1]]
        # Label the key phrases of interest in the current sentence with *.
        sentence[kp_indices_input[sentence_i,1]] += '*'
        sentence[kp_indices_input[sentence_i,0]] += '*'
        print('%s\n%s\t\t%s\t'
              % (current_key_phrase_pair,' '.join(sentence), str(actual_output_softmax[sentence_i])))
        user_input = raw_input('In your opinion, what should be the category of the key phrase pair? '
                                   'Please enter 1, 2, or 3. Enter 4 to stop answering.\n'
                                   '1. A is-a B\n2. B is-a A\n3. Neither.')
        user_input = util.get_valid_user_input(user_input, 1, 4)

        if user_input != 4:
            user_label_result = np.array([0,0,0])
            user_label_result[user_input-1] = 1
            user_label_results.append(user_label_result)
            num_user_labeled += 1

    actual_output_entropy_indices = actual_output_entropy_argsort[:num_user_labeled]

    if len(user_label_results) > 0:
        labeled_data, labeled_result = util.read_data_labeled_part(source_path, target_path, restore_param['sent_len'], shuffle=False)
        user_label_results = np.array(user_label_results)
        save_additional_label(unlabeled_data, actual_output_entropy_indices, user_label_results,labeled_data,labeled_result, source_path, target_path)

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
    tf.app.run()

