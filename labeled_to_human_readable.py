"""
This file contains functions used for active learning.
"""


from datetime import datetime
import os
import csv
import tensorflow as tf
import numpy as np

import util
import preprocessing_util
from label import label

NUM_TO_READ = 1000

def save_additional_label(unlabeled_data, additional_label_index, additional_label_result, labeled_data, labels, source_path,
                          target_path):
    if isinstance(additional_label_index,int):
        additional_label_index = [additional_label_index]
    additional_label_index_set = set(additional_label_index)
    num_unlabeled = unlabeled_data.shape[0]
    indices_not_included = [i for i in range(num_unlabeled) if i not in additional_label_index_set]
    data_x = np.concatenate((labeled_data, unlabeled_data[additional_label_index,...], unlabeled_data[indices_not_included,...]),axis=0)
    data_y  = np.concatenate((labels, additional_label_result), axis=0)

    np.savetxt(source_path, data_x, delimiter=' ', fmt='%d')
    util.save_labels_file(data_y,target_path)

def main(argv):
    restore_param = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir
    if argv is not None:
        source_path = argv[1]
        target_path = argv[2]
    if source_path is None:
        source_path = os.path.join(restore_param['data_dir'], 'test_cs_unlabeled_data_combined.txt')
    if target_path is None:
        target_path = os.path.join(restore_param['data_dir'], 'test_cs_labels_combined.txt')
    vocab_path = os.path.join(restore_param['data_dir'], 'test_cs_vocab_combined')

    labeled_data, labeled_result = util.read_data_labeled_part(source_path, target_path, restore_param['sent_len'],
                                                               shuffle=False)
    labeled_data = np.array(labeled_data)
    labeled_result = np.array(labeled_result)
    data_size = labeled_data.shape[0]

    # # Now hard code to take the first 1000
    # data_first_1000 = unlabeled_data

    sentence_indices_input = labeled_data[:,:-2]
    _,rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)
    sentence_input = preprocessing_util.indices_to_sentences(sentence_indices_input,rev_vocab)

    kp_indices_input = labeled_data[:,-2:]
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

    with open(os.path.join(FLAGS.train_dir, 'labeled_dataset_human.csv'),'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Key phrase pair(separated by one space)','Sentence(Key phrase labeled with *)','(Label)A is-a B','B is-a A','Neither'])
        for sentence_i in range(data_size):
            sentence = sentence_input[sentence_i]

            current_key_phrase_pair = sentence[kp_indices_input[sentence_i,0]] + ' ' + sentence[kp_indices_input[sentence_i,1]]
            # Label the key phrases of interest in the current sentence with *.
            sentence[kp_indices_input[sentence_i,1]] += '*'
            sentence[kp_indices_input[sentence_i,0]] += '*'
            csv_writer.writerow([current_key_phrase_pair,' '.join(sentence)] + labeled_result[sentence_i,...].tolist())

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
    tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'sanity_check_co_training/1491833701'), 'Directory of the checkpoint files')
    tf.app.run(main=main, argv=("","sanity_check_co_training/1491833701/test_cs_unlabeled_data_combined_round_4.txt",
                                "sanity_check_co_training/1491833701/test_cs_labels_combined_round_4.txt"))

