"""
This file contains functions used for active learning online.
"""


from datetime import datetime
import os
from shutil import copy2
import tensorflow as tf
import numpy as np

import util
import preprocessing_util
from label import label
from train import train
from active_learning_offline_util import save_additional_label

class ActiveLabelingUtilOnline:
    def __init__(self,cnn_model_path,source_path,target_path,vocab_path,sent_len,labeled_save_dir):
        """

        :param cnn_model_path: Path to a trained cnn model.
        :param source_path: Path to instance data, the latter part of which will be labeled during active learning.
        :param target_path: Path to labels for already labeled part of the data.
        :param vocab_path: Path to vocab file.
        :param labeled_save_dir: Directory to which the labeled files will be stored.
        """
        unlabeled_data = util.read_data_unlabeled_part(source_path, target_path, sent_len, shuffle=False)
        self.unlabeled_data = np.array(unlabeled_data)
        self.data_size = self.unlabeled_data.shape[0]


        self.labeled_data, self.labeled_result = util.read_data_labeled_part(source_path, target_path, sent_len, shuffle=False)

        sentence_indices_input = self.unlabeled_data[:, :-2]
        self.vocab_path = vocab_path
        _, rev_vocab = preprocessing_util.initialize_vocabulary(vocab_path)
        self.sentence_input = preprocessing_util.indices_to_sentences(sentence_indices_input, rev_vocab)
        self.kp_indices_input = self.unlabeled_data[:, -2:]

        for i, sentence in enumerate(self.sentence_input):
            # Label the key phrases of interest in the current sentence with *.
            sentence[self.kp_indices_input[i, 0]] += '*'
            sentence[self.kp_indices_input[i, 1]] += '*'

        self.update_labeled_save_dir(labeled_save_dir)
        # self.labeled_save_dir = labeled_save_dir
        # self.source_save_dir = os.path.join(labeled_save_dir, 'test_cs_unlabeled_data_combined.txt')
        # self.target_save_dir = os.path.join(labeled_save_dir, 'test_cs_labels_combined.txt')
        # self.vocab_save_dir = os.path.join(labeled_save_dir, 'test_cs_vocab_combined')

        label_config = util.load_from_dump(os.path.join(cnn_model_path, 'flags.cPickle'))
        label_config['train_dir']=cnn_model_path
        _, predicted_label = label(self.unlabeled_data,config=label_config)

        assert predicted_label.shape[0] == self.data_size

        predicted_label_exp = np.exp(predicted_label)
        predicted_label_softmax = predicted_label_exp / np.sum(predicted_label_exp, axis=1, keepdims=True)
        # Entropy = -sum(p * log p) so this is actually the negative of entropy. For sorting purpose I took out the neg.
        predicted_label_entropy = np.sum(np.multiply(predicted_label_softmax, np.log(predicted_label_softmax)), axis=1)

        # The following are ways to rank what question should be asked first.
        # The first one uses entropy, but there might be some implementation errors.
        self.predicted_label_entropy_argsort = np.argsort(predicted_label_entropy, axis=0).tolist()

        pass

    def database_to_labeled(self,database):
        """
        Process the database and convert it into labeled data. The labeled data is then saved in path provided during
        initialization.
        :param database:
        :return: None
        """
        #
        instance_labels = {}  # Key = instance id (as index in dataset) and value = list of labels.
        for user in database.usernames:
            instances, labels = database.get_user_instance_and_labels(user)
            relevant_instance_ids, relevant_instance_indices = self.get_relevant_instance_ids(instances)
            for i, relevant_instance_id in enumerate(relevant_instance_ids):
                if relevant_instance_id in instance_labels:
                    instance_labels[relevant_instance_id] = instance_labels[relevant_instance_id] + labels[i]
                else:
                    instance_labels[relevant_instance_id] = labels[i]

        additional_label_index = []
        additional_label_result = []
        for index, result in instance_labels.iteritems():
            majority = np.zeros((3))
            result_argmax= np.argmax(result[:3]) # Ignore the last column because it represents 'not sure/skip'.

            # If there are equal number of people who thinks that there is a relationship and who thinks that there is
            # not, set the result to "Neither" or no relation to be safe.
            if result[result_argmax] == result[2]:
                majority[2] = 1
            else:
                majority[result_argmax] = 1

            additional_label_index.append(index)
            additional_label_result.append(majority)

        save_additional_label(self.unlabeled_data, additional_label_index, additional_label_result, self.labeled_data,
                              self.labeled_result, self.source_save_dir, self.target_save_dir)
        copy2(self.vocab_path, self.vocab_save_dir)
        print("Additional labels saved")

    def get_new_unlabeled_instance(self,user_labeled_instance_ids):
        """
        Give a new unlabeled instance as well as its id that did not appear in the list of user_labeled_instance_ids.
        :param user_labeled_instance_ids:
        :return: (unlabeled instance key phrase pair, unlabeled instance sentence, instance id)
        """
        # Get set of ids relevant to the current set of data.
        relevant_instance_ids, _ = self.get_relevant_instance_ids(user_labeled_instance_ids)
        relevant_instance_ids = set(relevant_instance_ids)

        for index_to_be_labeled in self.predicted_label_entropy_argsort:
            if index_to_be_labeled not in relevant_instance_ids:
                sentence = self.sentence_input[index_to_be_labeled]
                current_key_phrase_pair = sentence[self.kp_indices_input[index_to_be_labeled, 0]].strip('*') + ' ' + sentence[
                    self.kp_indices_input[index_to_be_labeled, 1]].strip('*')
                sentence = ' '.join(sentence)
                current_instance_id = self.labeled_save_dir + ',' + str(index_to_be_labeled)
                return (current_key_phrase_pair,sentence,current_instance_id)
        print('Running out of instance to label! This is pretty unusual.')
        return None

    def get_relevant_instance_ids(self,user_labeled_instance_ids):
        """

        :param user_labeled_instance_ids: ids in format: 'labeled_save_dir' + ',' + 'index_in_unlabeled_data'
        :return: a SET of instance ids that has the same save_dir as this class's.
        """
        ret = []
        ret_indices = []
        for i, instance_id in enumerate(user_labeled_instance_ids):
            try:
                labeled_save_dir, index_in_unlabeled_data = instance_id.split(',')
                if labeled_save_dir == self.labeled_save_dir:
                    ret.append(int(index_in_unlabeled_data))
                    ret_indices.append(i)
            except:
                raise AssertionError("Incorrect instance id format. The format should be "
                                     "'labeled_save_dir' + ',' + 'index_in_unlabeled_data'. It is: %s" %(instance_id))
        return ret, ret_indices

    def instance_id_to_sentence(self, instance_id):
        try:
            labeled_save_dir, index_in_unlabeled_data = instance_id.split(',')
            if labeled_save_dir != self.labeled_save_dir:
                print('labeled_save_dir is not the newest. Received: %s, Newest: %s. Returning None'
                      %(labeled_save_dir, self.labeled_save_dir))
                return None
            index = int(index_in_unlabeled_data)
        except:
            raise AssertionError("Incorrect instance id format. The format should be "
                                 "'labeled_save_dir' + ',' + 'index_in_unlabeled_data'. It is: %s" % (instance_id))
        if index < 0 or index >= self.data_size :
            raise AssertionError("Incorrect index. It should be non negative and smaller than data size. It is now %d"
                                 %(index))
        return self.sentence_input[index]

    def update_labeled_save_dir(self,labeled_save_dir):

        self.labeled_save_dir = labeled_save_dir
        self.source_save_dir = os.path.join(labeled_save_dir, 'test_cs_unlabeled_data_combined.txt')
        self.target_save_dir = os.path.join(labeled_save_dir, 'test_cs_labels_combined.txt')
        self.vocab_save_dir = os.path.join(labeled_save_dir, 'test_cs_vocab_combined')

class CNNTrainerOnline:
    def __init__(self,labeled_save_dir,cnn_model_save_dir,sent_len):
        self.source_path = os.path.join(labeled_save_dir, 'test_cs_unlabeled_data_combined.txt')
        self.target_path = os.path.join(labeled_save_dir, 'test_cs_labels_combined.txt')
        self.sent_len = sent_len
        train_size = 1000000
        self.train_data, self.test_data = util.read_data(self.source_path, self.target_path, self.sent_len,
                                               attention_path=None, train_size=train_size)
    def start_train(self):
        # TODO: train uses lots of flags. Should I refactor the code? Or should I just call os?  If I call os, I don't
        # even need to load the train_data.
        train(self.train_data, self.test_data)


