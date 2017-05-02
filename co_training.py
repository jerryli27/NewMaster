from datetime import datetime
import hashlib
import time
import os
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import cnn
import util
import multilayer

import train
import label
import preprocessing_util
import train_kp_pair_classifier
from active_learning_offline_util import save_additional_label
from infer_labels_util import infer_from_labeled
from util import save_labels_file

FLAGS = tf.app.flags.FLAGS
CATEGORY_NAME = ["A is-a B", "B is-a A", "Neither"]

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

def draw_from_unused_unlabeled(unlabeled_data, used_unlabeled_kp_pair_set, num_to_draw):
    # Return the drawn data and the new set of used_unlabeled_kp_pair_set
    assert isinstance(used_unlabeled_kp_pair_set, set)
    num_unlabeled_data = unlabeled_data.shape[0]
    all_indices = set([i for i in range(num_unlabeled_data)])
    random_indices = random.sample(all_indices, len(all_indices))
    drawn_data = []
    drawn_indices = []
    current_used_unlabeled_kp_pair_set = set()
    i = 0
    while len(drawn_data) < num_to_draw:
        current_kp_pair_indices = unlabeled_data[random_indices[i],-2:]
        current_kp_pair = (unlabeled_data[random_indices[i],current_kp_pair_indices[0]],unlabeled_data[random_indices[i],current_kp_pair_indices[1]])
        if current_kp_pair not in used_unlabeled_kp_pair_set:
            current_used_unlabeled_kp_pair_set.add(current_kp_pair)
            drawn_data.append(unlabeled_data[random_indices[i]])
            drawn_indices.append(random_indices[i])
        i += 1
    drawn_data = np.array(drawn_data)
    drawn_indices = np.array(drawn_indices)
    used_unlabeled_kp_pair_set = used_unlabeled_kp_pair_set.union(current_used_unlabeled_kp_pair_set)
    return drawn_data, used_unlabeled_kp_pair_set, drawn_indices


def draw_from_unlabeled(unlabeled_data, num_to_draw):
    # Return the drawn data
    num_unlabeled_data = unlabeled_data.shape[0]
    unused_indices = set([i for i in range(num_unlabeled_data)])
    drawn_indices = random.sample(unused_indices, num_to_draw)
    drawn_data = unlabeled_data[drawn_indices]
    return drawn_data, drawn_indices

def check_conflict_and_merge(additional_label_index_list, additional_label_result_list):
    ret_additional_label_index = []
    ret_additional_label_result = []

    for i in range(len(additional_label_index_list)):

        k = 0
        while k < len(additional_label_index_list[i]):
            found_conflict = False
            for j in range(i+1, len(additional_label_index_list)):
                # K is the actual index. i and j are indices of the number of lists given.
                if additional_label_index_list[i][k] in additional_label_index_list[j]:
                    j_index = additional_label_index_list[j].index(additional_label_index_list[i][k])
                    if np.any(additional_label_result_list[i][k] != additional_label_result_list[j][j_index]):
                        print("There are conflicts in the labels of classifier %d #%d and classifier %d #%d. "
                                  "The result %s is different from %s. Excluding the two from the output."
                                  %(i,k,j,j_index,
                                    str(additional_label_result_list[i][k]),
                                    str(additional_label_result_list[j][j_index])))
                        additional_label_index_list[i].pop(k)
                        additional_label_index_list[j].pop(j_index)
                        additional_label_result_list[i].pop(k)
                        additional_label_result_list[j].pop(j_index)
                        found_conflict = True
                        break
                    else:
                        # Get rid of the duplicate item.
                        additional_label_index_list[j].pop(j_index)
                        additional_label_result_list[j].pop(j_index)
            if not found_conflict:
                ret_additional_label_index.append(additional_label_index_list[i][k])
                ret_additional_label_result.append(additional_label_result_list[i][k])
                k += 1

    return ret_additional_label_index, ret_additional_label_result


def compute_product_and_save(additional_label_result_list, latest_checkpoint_dir, sentence_input, kp_indices_input):
    """

    :param additional_label_result_list: a list of numpy arrays with shape (FLAGS.test_size_per_round, num_labels)
    :return:
    """
    num_classifiers = len(additional_label_result_list)
    product = np.product(np.array(additional_label_result_list), axis=0)
    sentence_i_list = np.argsort(-np.max(product, axis=1)).tolist()
    product_argmax_argmax = np.argmax(product, axis=1)
    ret_additional_label_index = []
    ret_additional_label_result = []

    with open(os.path.join(latest_checkpoint_dir, 'probability_product_added_instances.tsv'), "w") as inferred_instances_f:

        inferred_instances_f.write('Type\tSentence\t\tProbability [A is-a B, B is-a A, Neither]\n')
        additional_label_num_positive = 0
        additional_label_num_negative = 0
        for sentence_i in sentence_i_list:
            # # This is the current max probability
            # current_softmax = actual_output_softmax[sentence_i,actual_output_argmax[sentence_i]]
            sentence = sentence_input[sentence_i]
            # Label the key phrases of interest in the current sentence with *.
            sentence[kp_indices_input[sentence_i, 1]] += '*'
            sentence[kp_indices_input[sentence_i, 0]] += '*'
            if product_argmax_argmax[sentence_i] == 2:
                current_type = 'Neither'
                if additional_label_num_negative >= FLAGS.co_training_has_relation_num_label_negative * num_classifiers:
                    continue
                else:
                    additional_label_num_negative += 1
            if product_argmax_argmax[sentence_i] == 0:
                current_type = 'A is-a B'
                if additional_label_num_positive >= FLAGS.co_training_has_relation_num_label_positive * num_classifiers:
                    continue
                else:
                    additional_label_num_positive += 1
            elif product_argmax_argmax[sentence_i] == 1:
                current_type = 'B is-a A'
                if additional_label_num_positive >= FLAGS.co_training_has_relation_num_label_positive * num_classifiers:
                    continue
                else:
                    additional_label_num_positive += 1

            inferred_instances_f.write('%s\t%s\t\t%s\n'
                                       % (current_type, ' '.join(sentence), str(product[sentence_i])))


            ret_additional_label_index.append(sentence_i)
            # If use_product_method is off, then the result is the label.
            current_additional_label_result = np.zeros((3,))
            current_additional_label_result[product_argmax_argmax[sentence_i]] = 1
            ret_additional_label_result.append(current_additional_label_result)
            if additional_label_num_positive >= FLAGS.co_training_has_relation_num_label_positive * num_classifiers and \
                            additional_label_num_negative >= FLAGS.co_training_has_relation_num_label_negative * num_classifiers:
                break
        print("Number of additional data points added through combining both classifiers by taking the product"
              ": %d positives and %d negatives out of %d unlabeled instances."
              % (additional_label_num_positive, additional_label_num_negative, len(sentence_i_list)))
    return ret_additional_label_index, ret_additional_label_result

def cross_validation_split(source_path,target_path,save_dir, fold_number = 10):
    labeled_data, labeled_result = util.read_data_labeled_part(source_path, target_path, FLAGS.sent_len,
                                                               shuffle=False)
    unlabeled_data = util.read_data_unlabeled_part(source_path, target_path, FLAGS.sent_len, shuffle=False,
                                                   hide_key_phrases=False)
    num_labeled = labeled_data.shape[0]
    assert fold_number <= num_labeled and isinstance(fold_number, int)
    random_indices = np.random.permutation(num_labeled)
    for i in range(fold_number):
        start = i * num_labeled / fold_number
        end = (i + 1) * num_labeled / fold_number

        # Save picked instances
        np.savetxt(os.path.join(save_dir,"cross_validation_val_%d_data.txt"%(i)), labeled_data[random_indices[start:end]], delimiter=' ', fmt='%d')
        # Save labels
        save_labels_file(labeled_result[random_indices[start:end]], os.path.join(save_dir,"cross_validation_val_%d_labels.txt"%(i)))
        # Then save other instances
        np.savetxt(os.path.join(save_dir,"cross_validation_train_%d_data.txt"%(i)), np.concatenate((labeled_data[np.concatenate((random_indices[:start], random_indices[end:]))], unlabeled_data)), delimiter=' ', fmt='%d')
        # Save their labels
        save_labels_file(labeled_result[np.concatenate((random_indices[:start], random_indices[end:]))], os.path.join(save_dir,"cross_validation_train_%d_labels.txt"%(i)))
    print("%d-fold cross validation data generation complete." %(fold_number))

def main(argv=None):
    # Flags are defined in train.py
    if FLAGS.hide_key_phrases:
        raise AssertionError("Please turn the hide_key_phrases off for co-training.")

    # First generate cross validation data if it does not exist.

    if not os.path.exists(FLAGS.cross_validation_dir):
        print("Cross validation data folder does not exist. Creating one.")
        os.mkdir(FLAGS.cross_validation_dir)
        source_path = os.path.join(FLAGS.data_dir, 'test_cs_unlabeled_data_combined_inferred.txt')
        target_path = os.path.join(FLAGS.data_dir, 'test_cs_labels_combined_inferred.txt')
        cross_validation_split(source_path, target_path, FLAGS.cross_validation_dir, fold_number=FLAGS.cross_validation_fold)

    for cross_val_round_i in range(FLAGS.cross_validation_fold):

        if not os.path.exists(FLAGS.train_dir):
            os.mkdir(FLAGS.train_dir)
        latest_sentence_checkpoint_dir = None
        latest_pair_checkpoint_dir = None
        latest_checkpoint_dir = None
        used_unlabeled_kp_pair_set = set()
        # The validation set is separate from the test and training set from the very beginning.
        val_source_path = os.path.join(FLAGS.cross_validation_dir,"cross_validation_val_%d_data.txt"%(cross_val_round_i))
        val_target_path = os.path.join(FLAGS.cross_validation_dir,"cross_validation_val_%d_labels.txt"%(cross_val_round_i))
        val_labeled_data, val_labeled_result = util.read_data_labeled_part(val_source_path, val_target_path, FLAGS.sent_len,
                                                                   shuffle=False)
        # For legacy code reasons, I have to add a None column to the training data...
        val_data = np.array(zip(val_labeled_data, val_labeled_result, [None] * val_labeled_result.shape[0]))
        val_precision = []
        val_recall = []
        val_pr_auc = []  # Precision recall area under the curve.

        for round_i in range(FLAGS.max_co_training_rounds):
            # load dataset

            if round_i == 0:

                source_path = os.path.join(FLAGS.cross_validation_dir, "cross_validation_train_%d_data.txt" % (cross_val_round_i))
                target_path = os.path.join(FLAGS.cross_validation_dir, "cross_validation_train_%d_labels.txt" % (cross_val_round_i))
                # source_path = os.path.join(FLAGS.data_dir, 'test_cs_unlabeled_data_combined_inferred_train.txt')
                # target_path = os.path.join(FLAGS.data_dir, 'test_cs_labels_combined_inferred_train.txt')
            else:
                source_path = os.path.join(latest_checkpoint_dir,
                                           'test_cs_unlabeled_data_combined_round_%d.txt' % (round_i))
                target_path = os.path.join(latest_checkpoint_dir, 'test_cs_labels_combined_round_%d.txt' % (round_i))
            train_data, test_data = util.read_data(source_path, target_path, FLAGS.sent_len,
                                                   attention_path=None, train_size=FLAGS.train_size,
                                                   hide_key_phrases=False)
            # I probably need to implement getting all the sentences with the same kp here as well?
            train_data_hide_kp, test_data_hide_kp = util.read_data(source_path, target_path, FLAGS.sent_len,
                                                   attention_path=None, train_size=FLAGS.train_size,
                                                   hide_key_phrases=True)

            print("Round %d. Reading labeled data from previous round." %(round_i))
            labeled_data, labeled_result = util.read_data_labeled_part(source_path, target_path, FLAGS.sent_len,
                                                                       shuffle=False)
            unlabeled_data = util.read_data_unlabeled_part(source_path, target_path, FLAGS.sent_len, shuffle=False,
                                                           hide_key_phrases=False)
            unlabeled_data_hide_kp = util.read_data_unlabeled_part(source_path, target_path, FLAGS.sent_len, shuffle=False,
                                                                   hide_key_phrases=True)

            # For each round, we draw a fresh set of unlabeled data and label them using the trained classifier.
            current_unlabeled_data, used_unlabeled_kp_pair_set, current_drawn_indices = draw_from_unused_unlabeled(unlabeled_data,
                                                                                             used_unlabeled_kp_pair_set,
                                                                                             FLAGS.test_size_per_round)
            current_unlabeled_data_hide_kp = [unlabeled_data_hide_kp[i] for i in current_drawn_indices]
            # Currently this one works, but we need a version that throws away used ones. So we need to keep track of which
            # ones we've used.
            # current_unlabeled_data, current_drawn_indices = draw_from_unlabeled(unlabeled_data,
            #                                                                                  FLAGS.test_size_per_round)
            # current_unlabeled_data_hide_kp = [unlabeled_data_hide_kp[i] for i in current_drawn_indices]



            additional_label_index = []
            additional_label_result = []

            for classifier_i in range(2):
                additional_label_index.append([])
                additional_label_result.append([])
                if _is_sentence_train(classifier_i):
                    train.train(train_data_hide_kp, test_data_hide_kp)
                    latest_sentence_checkpoint_dir = util.get_latest_checkpoint_dir(FLAGS.train_dir)
                else:
                    train_kp_pair_classifier.train(train_data,test_data)
                    latest_pair_checkpoint_dir = util.get_latest_checkpoint_dir(FLAGS.train_dir)

                # Refresh the latest checkpoint.
                latest_checkpoint_dir = util.get_latest_checkpoint_dir(FLAGS.train_dir)
                restore_param = util.load_from_dump(os.path.join(latest_checkpoint_dir, 'flags.cPickle'))
                restore_param['train_dir'] = latest_checkpoint_dir
                if _is_sentence_train(classifier_i):
                    x_input, actual_output = label.label(current_unlabeled_data_hide_kp, restore_param)
                else:
                    x_input, actual_output = train_kp_pair_classifier.label(current_unlabeled_data, restore_param)


                actual_output_exp = np.exp(actual_output)
                actual_output_softmax = actual_output_exp / np.sum(actual_output_exp, axis=1, keepdims=True)
                actual_output_argmax = np.argmax(actual_output_softmax, axis=1)
                # If we do not want "Neither" relation, then calculate max on only the first 2 dimensions.
                # sentence_i_list = np.argsort(-np.max(actual_output_softmax[..., :2], axis=1)).tolist()
                if FLAGS.use_product_method:
                    sentence_i_list = range(actual_output_softmax.shape[0])
                else:
                    sentence_i_list = np.argsort(-np.max(actual_output_softmax, axis=1)).tolist()

                # We need the version with key phrases not replaced in order to print things correctly.
                sentence_indices_input = current_unlabeled_data[:, :-2]
                vocab_path = os.path.join(restore_param['data_dir'], 'test_cs_vocab_combined')
                _, rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)
                sentence_input = preprocessing_util.indices_to_sentences(sentence_indices_input, rev_vocab, ignore_pad=True)

                kp_indices_input = current_unlabeled_data[:, -2:]

                with open(os.path.join(latest_checkpoint_dir, 'added_instances.tsv'), "w") as inferred_instances_f:

                    inferred_instances_f.write('Type\tSentence\t\tProbability [A is-a B, B is-a A, Neither]\n')
                    additional_label_num_positive = 0
                    additional_label_num_negative = 0
                    for sentence_i in sentence_i_list:
                        # # This is the current max probability
                        # current_softmax = actual_output_softmax[sentence_i,actual_output_argmax[sentence_i]]
                        sentence = sentence_input[sentence_i]
                        # Label the key phrases of interest in the current sentence with *.
                        sentence[kp_indices_input[sentence_i, 1]] += '*'
                        sentence[kp_indices_input[sentence_i, 0]] += '*'
                        if actual_output_argmax[sentence_i] == 2:
                            current_type = 'Neither'
                            if not FLAGS.use_product_method and additional_label_num_negative >= FLAGS.co_training_has_relation_num_label_negative:
                                continue
                            else:
                                additional_label_num_negative += 1
                        if actual_output_argmax[sentence_i] == 0:
                            current_type = 'A is-a B'
                            if not FLAGS.use_product_method and additional_label_num_positive >= FLAGS.co_training_has_relation_num_label_positive:
                                continue
                            else:
                                additional_label_num_positive += 1
                        elif actual_output_argmax[sentence_i] == 1:
                            current_type = 'B is-a A'
                            if not FLAGS.use_product_method and additional_label_num_positive >= FLAGS.co_training_has_relation_num_label_positive:
                                continue
                            else:
                                additional_label_num_positive += 1

                        inferred_instances_f.write('%s\t%s\t\t%s\n'
                                                % (current_type, ' '.join(sentence), str(actual_output_softmax[sentence_i])))


                        if not FLAGS.use_product_method:
                            additional_label_index[classifier_i].append(sentence_i)
                            # If use_product_method is off, then the result is the label.
                            current_additional_label_result = np.zeros((3,))
                            current_additional_label_result[actual_output_argmax[sentence_i]] = 1
                            additional_label_result[classifier_i].append(current_additional_label_result)
                            if additional_label_num_positive >= FLAGS.co_training_has_relation_num_label_positive and \
                                additional_label_num_negative >= FLAGS.co_training_has_relation_num_label_negative:
                                break
                        else:
                            # If use_product_method is on, then the result is the output softmax, i.e. probability.
                            current_additional_label_result = actual_output_softmax[sentence_i]
                            additional_label_result[classifier_i].append(current_additional_label_result)

                print("Number of additional data points added through co-training classifier %d"
                      ": %d positives and %d negatives out of %d unlabeled instances."
                      %(classifier_i ,additional_label_num_positive, additional_label_num_negative, len(sentence_i_list)))

            # Check if there are any conflicts and merge the additional labels labeled by the two classifier.
            if not FLAGS.use_product_method:
                merged_additional_label_index, merged_additional_label_result = check_conflict_and_merge(additional_label_index,additional_label_result)
            else:
                merged_additional_label_index, merged_additional_label_result = compute_product_and_save(additional_label_result, latest_checkpoint_dir, sentence_input, kp_indices_input)

            latest_checkpoint_dir = util.get_latest_checkpoint_dir(FLAGS.train_dir)
            save_source_path = os.path.join(latest_checkpoint_dir, 'test_cs_unlabeled_data_combined_round_%d.txt' % (round_i + 1))
            save_target_path = os.path.join(latest_checkpoint_dir, 'test_cs_labels_combined_round_%d.txt' % (round_i + 1))
            # Now recover the original index in the unlabeled data.
            merged_additional_label_index = [current_drawn_indices[i] for i in merged_additional_label_index]
            # Save the additionally labeled 2p+2n examples.
            save_additional_label(unlabeled_data, merged_additional_label_index, merged_additional_label_result,
                                  labeled_data, labeled_result, save_source_path, save_target_path)

            # I also need to get rid of those inferred instances from the whole bag of unlabeled dataset that we're drawing
            # from at each round.
            before_inference_unlabeled_data = util.read_data_unlabeled_part(save_source_path, save_target_path, FLAGS.sent_len, shuffle=False)
            inferred_additional_label_index, inferred_additional_label_result = infer_from_labeled(save_source_path,
                                                                                                   save_target_path,
                                                                                                   FLAGS.sent_len, vocab_path,
                                                                                                   do_save=True,
                                                                                                   save_source_path=save_source_path,
                                                                                                   save_target_path=save_target_path)
            inferred_additional_data = before_inference_unlabeled_data[inferred_additional_label_index]
            inferred_additional_sentence_index = inferred_additional_data[:,:-2]
            inferred_additional_kp_index = inferred_additional_data[:,-2:]
            inferred_additional_sentence_input = preprocessing_util.indices_to_sentences(inferred_additional_sentence_index, rev_vocab, ignore_pad=True)

            inferred_additional_label_result_argmax = np.argmax(inferred_additional_label_result, axis=1)
            with open(os.path.join(latest_checkpoint_dir, 'inferred_instances.tsv'), "w") as inferred_instances_f:
                inferred_instances_f.write('Type\tSentence\n')

                for sentence_i in range(inferred_additional_kp_index.shape[0]):
                    # # This is the current max probability
                    # current_softmax = actual_output_softmax[sentence_i,actual_output_argmax[sentence_i]]
                    sentence = inferred_additional_sentence_input[sentence_i]
                    # Label the key phrases of interest in the current sentence with *.
                    sentence[inferred_additional_kp_index[sentence_i, 1]] += '*'
                    sentence[inferred_additional_kp_index[sentence_i, 0]] += '*'
                    if inferred_additional_label_result_argmax[sentence_i] == 2:
                        current_type = 'Neither'
                    if inferred_additional_label_result_argmax[sentence_i] == 0:
                        current_type = 'A is-a B'
                    elif inferred_additional_label_result_argmax[sentence_i] == 1:
                        current_type = 'B is-a A'
                    inferred_instances_f.write('%s\t%s\n' % (current_type, ' '.join(sentence)))

            # Now all is left is to use the validation dataset to calculate the area under precision recall curve.
            val_precision.append([[[] for _ in range(3)] for _ in range(3)])
            val_recall.append([[[] for _ in range(3)] for _ in range(3)])
            val_pr_auc.append([[0.0, 0.0, 0.0] for _ in range(3)])
            # Each time we calculate the precision recall for classifier 1, 2, and combined.
            for classifier_j in range(3):
                if classifier_j == 0:
                    # Use classifier 1.
                    restore_param = util.load_from_dump(os.path.join(latest_sentence_checkpoint_dir, 'flags.cPickle'))
                    restore_param['train_dir'] = latest_sentence_checkpoint_dir
                    _, val_actual_output = label.label(val_labeled_data, restore_param)
                elif classifier_j == 1:
                    # Use classifier 2.
                    restore_param = util.load_from_dump(os.path.join(latest_pair_checkpoint_dir, 'flags.cPickle'))
                    restore_param['train_dir'] = latest_pair_checkpoint_dir
                    _, val_actual_output = train_kp_pair_classifier.label(val_labeled_data, restore_param)
                else:
                    # Use both classifier and, due to design choice of caring more about precision than recall, label
                    # an instance as having a subcategory relation only when both classifier agrees, otherwise output
                    # no relation, aka `Neither`.
                    restore_param = util.load_from_dump(os.path.join(latest_sentence_checkpoint_dir, 'flags.cPickle'))
                    restore_param['train_dir'] = latest_sentence_checkpoint_dir
                    _, val_actual_output_sentence = label.label(val_labeled_data, restore_param)
                    restore_param = util.load_from_dump(os.path.join(latest_pair_checkpoint_dir, 'flags.cPickle'))
                    restore_param['train_dir'] = latest_pair_checkpoint_dir
                    _, val_actual_output_pair = train_kp_pair_classifier.label(val_labeled_data, restore_param)
                    val_actual_output_sentence_argmax = np.argmax(val_actual_output_sentence, axis=1)
                    val_actual_output_pair_argmax = np.argmax(val_actual_output_pair, axis=1)

                    # Label the actual output as [1,0,0] if both classify as A is B, [0,1,0] if both classify as B is A,
                    # and [0,0,1] in all other situations.
                    val_actual_output = np.array([[1 if k == val_actual_output_sentence_argmax[j] else 0 for k in range(3)]
                                                  if np.all(val_actual_output_sentence_argmax[j] == val_actual_output_pair_argmax[j])
                                                  else [0,0,1] for j in range(val_actual_output_sentence.shape[0])])

                val_actual_output_exp = np.exp(val_actual_output)
                val_actual_output_softmax = val_actual_output_exp / np.sum(val_actual_output_exp, axis=1, keepdims=True)
                for i in range(3):
                    val_precision[round_i][classifier_j][i], val_recall[round_i][classifier_j][i], _ = precision_recall_curve(val_labeled_result[:, i], val_actual_output_softmax[:, i])
                    val_pr_auc[round_i][classifier_j][i] = average_precision_score(val_labeled_result[:, i], val_actual_output_softmax[:, i], )

        # Lastly output the precision recall file for each classifier and each category.
        with open(os.path.join(latest_checkpoint_dir, 'pr_auc.tsv'), "w") as f:
            for classifier_j in range(3):
                for i in range(3):
                    f.write("Classifier%d_%s\t%s\n"
                            %(classifier_j, CATEGORY_NAME[i],
                              "\t".join([str(val_pr_auc[round_i][classifier_j][i])
                                         for round_i in range(FLAGS.max_co_training_rounds)])))

        np.save(os.path.join(latest_checkpoint_dir, 'precision_recall_data'),
                np.array([val_precision, val_recall, val_pr_auc]))

if __name__ == '__main__':
    cnn.define_flags()
    train.define_flags()
    multilayer.define_new_flags()
    # tf.app.flags.DEFINE_float('co_training_has_relation_prob_upper_threshold', 0.9,
    #                           "The upper threshold for the probability of belonging to a class. If the probability of "
    #                           "an unlabeled instance is above this threshold, it will be used as extra training data "
    #                           "for co training.")
    # tf.app.flags.DEFINE_float('co_training_no_relation_prob_upper_threshold', 0.9999,
    #                           "The threshold for the probability of belonging to no relation `Neither`. If the "
    #                           "probability of an unlabeled instance is above this threshold, it will be used as extra "
    #                           "training data for co training.")
    tf.app.flags.DEFINE_float('co_training_has_relation_num_label_positive', 2,
                              "For each round of co-training, we draw the top x positive example (instances with "
                              "relation other than `Neither` and add it to the labeled training examples.")
    tf.app.flags.DEFINE_float('co_training_has_relation_num_label_negative', 30,
                              "For each round of co-training, we draw the top x negative example (instances with "
                              "relation other than `Neither` and add it to the labeled training examples.")
    tf.app.flags.DEFINE_integer('max_co_training_rounds', 10,
                                "The maximum number of rounds for co training.")
    tf.app.flags.DEFINE_integer('test_size_per_round', 1000,
                                "The number of instance drawn from the unlabeled dataset per round for co training.")
    tf.app.flags.DEFINE_string('cross_validation_dir', 'cross_validation_data',
                                "The folder containing pre-split cross validation datasets.")
    tf.app.flags.DEFINE_integer('cross_validation_fold', 10,
                                "Do x-fold cross validation.")

    tf.app.flags.DEFINE_boolean('use_product_method', False,
                                "If True, the additional labels are picked based on the product of the estimated "
                                "probability. Otherwise the two classifiers pick the additional labels independent of "
                                "each other.")

    tf.app.run()