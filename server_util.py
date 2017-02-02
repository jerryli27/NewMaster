"""
This file contains utility function for running the active labeling server.
"""

import os
import pickle
import threading
import time


def load_pickle(path):
    """
    :param path:  Directory the data file is stored.
    :return: Database file stored at path.
    """
    if os.path.isfile(path):
        try:
            with open(path, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
        except:
            raise IOError("Failed to use pickle to read file at %s." %(path))
        return data
    else:
        print("Cannot find file at %s." %(path))
        return None

def save_pickle(path, data):
    """

    :param path: Directory to store the database file.
    :param data: Data to be saved.
    :return: None
    """
    try:
        with open(path, 'wb') as output:
            # Pickle dictionary using protocol 0.
            pickle.dump(data, output)
    except:
        raise IOError("Failed to use pickle to save file at %s." % (path))

def initialize_database(path):
    """
    Creates a database at path if a database does not already exists.
    :param path: Directory to store the database file.
    :return: The newly created database file, or the one already exists at path.
    """

    if os.path.isfile(path):
        return load_pickle(path)
    else:
        print("Database file does not exists at %s. Creating a new one." % (path))
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        database = Database()
        save_pickle(path, database)
        return database

class Database:
    def __init__(self):
        self.usernames = set()
        self.user_instance_ids = {}
        self.user_labels = {}
        self.user_comments = {}
        self.instance_id_to_sentence = {}
    def add_new_user(self,username):
        """
        :param username: The new user's username
        :return: True if user successfully added. False if user already exists.
        """
        if username in self.usernames:
            return False
        else:
            self.usernames.add(username)
            self.user_instance_ids[username] = []
            self.user_labels[username] = []
            self.user_comments[username] = []
            return True

    def get_user_num_labeled(self,username):
        if username in self.usernames:
            ret = len(self.user_instance_ids[username])
            if ret != len(self.user_labels[username]):
                raise AssertionError('For some reason, the instance ids list and the user labels list does not have '
                                     'the same length. In function get_user_num_labeled for user %s, the '
                                     'user_instance_id length is %d whereas the user_labels length is %d'
                                     % (username,ret,len(self.user_labels[username])))
            if ret != len(self.user_comments[username]):
                raise AssertionError('For some reason, the instance ids list and the user labels list does not have '
                                     'the same length. In function get_user_num_labeled for user %s, the '
                                     'user_instance_id length is %d whereas the user_comments length is %d'
                                     % (username,ret,len(self.user_comments[username])))
            return ret
        else:
            return -1  # Indicating there is no such user.

    def get_user_instance_and_labels(self,username):
        if username in self.usernames:
            instances = self.user_instance_ids[username]
            labels = self.user_labels[username]
            if len(instances) != len(labels):
                raise AssertionError('For some reason, the instance ids list and the user labels list does not have '
                                     'the same length. In function get_user_instance_and_labels for user %s, the '
                                     'user_instance_id length is %d whereas the user_labels length is %d'
                                     % (username,len(instances),len(labels)))

            return instances, labels
        else:
            return None  # Indicating there is no such user.


    def add_user_instance_and_label(self,username, instance_id, label, sentence, comment):
        if username in self.usernames:
            self.user_instance_ids[username].append(instance_id)
            self.user_labels[username].append(label)
            self.user_comments[username].append(comment)
            if len(self.user_instance_ids[username]) != len(self.user_labels[username]):
                raise AssertionError('For some reason, the instance ids list and the user labels list does not have '
                                     'the same length. In function get_user_instance_and_labels for user %s, the '
                                     'user_instance_id length is %d whereas the user_labels length is %d'
                                     % (username,len(self.user_instance_ids[username]),len(self.user_labels[username])))
            if len(self.user_instance_ids[username]) != len(self.user_comments[username]):
                raise AssertionError('For some reason, the instance ids list and the user labels list does not have '
                                     'the same length. In function get_user_instance_and_labels for user %s, the '
                                     'user_instance_id length is %d whereas the user_comments length is %d'
                                     % (username,len(self.user_instance_ids[username]),len(self.user_comments[username])))
            if instance_id not in self.instance_id_to_sentence:
                self.instance_id_to_sentence[instance_id] = sentence

            return True
        else:
            return False  # Indicating there is no such user.


