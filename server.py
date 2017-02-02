#!/usr/bin/env python

import CGIHTTPServer, SimpleHTTPServer, BaseHTTPServer
import SocketServer

import os, sys
import base64
import json
import time
import threading
import numpy as np

import argparse

from cgi import parse_header, parse_multipart
from urlparse import parse_qs
from io import open
from shutil import move, copy2

import server_util
import active_learning_online_util

from util import get_latest_checkpoint_dir, remove_earliest_checkpoint

class MyHandler(CGIHTTPServer.CGIHTTPRequestHandler):
    def __init__(self, req, client_addr, server):
        CGIHTTPServer.CGIHTTPRequestHandler.__init__(self, req, client_addr, server)

    def parse_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        pdict['boundary'] = str(pdict['boundary']).encode("utf-8")
        if ctype == 'multipart/form-data':
            postvars = parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(
                self.rfile.read(length),
                keep_blank_values=1)
        else:
            postvars = {}
        return postvars

    def do_POST(self):
        """
        Upon receiving a post request, there are two possibilities:
        One is the post is only submitting the username and asking for a new instance to label.
        The other is the post is submitting the username, the instance, and its label  and asking for a new instance.
        In both cases, the message to send back is the same: a new instance that the user did not label yet.
        :return:
        """
        form = self.parse_POST()
        success = True and not server_frozen
        error_message = '' if not server_frozen else 'Server is temporary out of service due to updates in database. Please come back and check within 1~10 minutes.'

        if success and "username" in form:
            username_str = form["username"][0]
            username_str = username_str.decode()
            is_new_user = database.add_new_user(username_str)
            print('Debug: parsed user name %s, is new user: %s' %(username_str, str(is_new_user)))

            if "instance_id" in form:
                if not ("label" in form):
                    success = False
                    error_message = 'Instance_id is in form but label is not. client side error.'

                # Now do something with the instance id.
                user_instance_id = form["instance_id"][0]
                user_label_str = form["label"][0]
                user_label = np.zeros((4))
                user_comment = form["comments"][0]
                user_num_labeled = 0
                try:
                    user_label_index = int(user_label_str)
                    assert user_label_index <= 3 and user_label_index >= 0
                except:
                    success = False
                    error_message = 'Wrong user_label_str format. It should be 0, 1, 2, or 3. It is now %s' \
                                    %(user_label_str)
                if success:
                    user_label[user_label_index] = 1
                    user_sentence = active_labeling_util.instance_id_to_sentence(user_instance_id)
                    if user_sentence is None:
                        # This indicates that the labeled_save_dir is out of date.
                        success = False
                        error_message = 'Database and active learning CNN has been updated. ' \
                                        'labeled_save_dir out of date. Please refresh the page and clear any browser ' \
                                        'cache if needed.'
                    else:
                        database.add_user_instance_and_label(username_str,user_instance_id, user_label, user_sentence, user_comment)
                        server_util.save_pickle(config['database_path'],database)
            else:
                if not ("label" not in form):
                    success = False
                    error_message = 'Instance_id is not in form but label is. client side error.'
                # The request is just asking for a new instance. Continue.
            if success:
                user_instance_and_labels = database.get_user_instance_and_labels(username_str)
                if user_instance_and_labels is None:
                    success = False
                    error_message = 'user_instance_and_labels returned None, indicating username not found in ' \
                                    'database. Server side error'
                else:
                    user_instance_ids = user_instance_and_labels[0]
                    new_tuple = active_labeling_util.get_new_unlabeled_instance(user_instance_ids)
                    user_num_labeled = database.get_user_num_labeled(username_str)
                    if new_tuple is not None:
                        current_key_phrase_pair, sentence, current_instance_id = new_tuple
                    else:
                        success = False
                        error_message = 'You have exhausted the dataset! This is really unusal... You have labeled %d instances.' %(user_num_labeled)

        if success:
            content = str( "{ \"message\":\"The command Completed Successfully\" , \"Status\":\"200 OK\" , \"success\":true ,"
                           " \"used\":%s , \"key_phrase_pair\":\"%s\", \"sentence\":\"%s\" , \"instance_id\":\"%s\" , \"user_num_labeled\":%s }"
                           %(str(args.gpu), current_key_phrase_pair, sentence, current_instance_id, user_num_labeled)).encode("UTF-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            content = str("{ \"message\":\"The command Completed Successfully\" , \"Status\":\"200 OK\",\"success\":false ,"
                          " \"used\":%s , \"error_message\":\"%s\"}"
                          % (str(args.gpu), error_message)).encode("UTF-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        return


def asynchronous_retrain():
    """
    Tasks to complete: 0. Freeze any input. Force refresh if needed by checking whether cnn_model_path is up to date.
    1. Save the database into labeled data. Copy vocab over. 2. create a new labeled_save_dir, change the config of the
    current active_labeling_util. Defreeze.
    3. Move the pretrained embedding to the new data folder and train a new CNN model using the new labeled data.
    4. update the cnn_model_path, source_path, target_path,
    vocab_path, and create new labeled_save_dir in the new config. Save the config.
    5. Load the new config and create a new active_labeling_util using the new config
    6. Swap out the old and replace with the new one.
    7. Remove earliest checkpoint if there are too many of them.


    :return:
    """
    print('Running asynchronous_retrain')
    # Step 0
    print('Step 0 in asynchronous_retrain')
    global server_frozen
    server_frozen = True
    # Step 1
    print('Step 1 in asynchronous_retrain')
    global database, active_labeling_util
    if database is None:
        database = server_util.initialize_database(config['database_path'])
    else:
        active_labeling_util.database_to_labeled(database)
    # Step 2
    print('Step 2 in asynchronous_retrain')
    # global config
    previous_labeled_save_dir = config['labeled_save_dir']
    config['labeled_save_dir'] = './active_labeling_database/' + str(time.time())
    if not os.path.exists(config['labeled_save_dir']):
        os.makedirs(config['labeled_save_dir'])
    else:
        raise AssertionError("Why would there be a folder with the same name as the current time at %s?"
                             %(config['labeled_save_dir']))
    active_labeling_util.update_labeled_save_dir(config['labeled_save_dir'])
    server_frozen = False
    # Step 3
    print('Step 3 in asynchronous_retrain')
    print(previous_labeled_save_dir)
    copy2(os.path.join(previous_labeled_save_dir, 'emb.npy'), os.path.join(config['labeled_save_dir'], 'emb.npy'))
    os.system('%s ~/PycharmProjects/NewMaster/train.py --use_pretrain=True --batch_size=100 '
              '--vocab_size=20933 --data_dir="%s"'%(args.python_path,previous_labeled_save_dir))
    # Step 4
    print('Step 4 in asynchronous_retrain')
    config['cnn_model_path'] = get_latest_checkpoint_dir('./checkpoints')
    config['source_path'] = os.path.join(previous_labeled_save_dir, 'test_cs_unlabeled_data_combined.txt')
    config['target_path'] = os.path.join(previous_labeled_save_dir, 'test_cs_labels_combined.txt')
    config['vocab_path'] = os.path.join(previous_labeled_save_dir, 'test_cs_vocab_combined')
    # global args
    server_util.save_pickle(args.config_path,config)
    # Step 5
    print('Step 5 in asynchronous_retrain')
    start_time = time.time()
    new_active_labeling_util = active_learning_online_util.ActiveLabelingUtilOnline(
        config['cnn_model_path'], config['source_path'], config['target_path'],config['vocab_path'],
        config['sent_len'], config['labeled_save_dir'])
    end_time = time.time()
    print('Finished database and active learning initialization. Took %.1f seconds' % (end_time - start_time))
    # Step 6
    print('Step 6 in asynchronous_retrain')
    server_frozen = True
    active_labeling_util = new_active_labeling_util
    server_frozen = False
    # Step 7
    print('Step 7 in asynchronous_retrain')
    remove_earliest_checkpoint('./checkpoints')

    # call f() again in 3600 seconds
    print('Done.')
    threading.Timer(3600, asynchronous_retrain).start()  # TODO: change back to 3600

parser = argparse.ArgumentParser(description='chainer line drawing colorization server')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--port', '-p', type=int, default=8000,
                    help='using port')
parser.add_argument('--host', '-ho', default='localhost',
                    help='using host. If want to be served on web, set it to blank string "".')
parser.add_argument('--config_path', '-cf', default='./active_labeling_database/config.pkl',
                    help='config_path')
parser.add_argument('--python_path', '-py', default='python',
                    help='python path. Default should work. If OS is giving you trouble, change to the absolute path '
                         'of where python is installed.')
args = parser.parse_args()

print 'GPU: {}'.format(args.gpu)

if os.path.isfile(args.config_path):
    config = server_util.load_pickle(args.config_path)
else:
    config = {'database_path':'./active_labeling_database/database.pkl',
              'cnn_model_path':'./checkpoints/1485596219/',
              'source_path':'./hand_label_context_tool/test_cs_unlabeled_data_combined.txt',
              'target_path':'./hand_label_context_tool/test_cs_labels_combined.txt',
              'vocab_path':'./hand_label_context_tool/test_cs_vocab_combined',
              'sent_len':128,
              'labeled_save_dir':('./active_labeling_database/' + str(time.time()))
              }
    if not os.path.exists(os.path.dirname(args.config_path)):
        os.makedirs(os.path.dirname(args.config_path))
    server_util.save_pickle(args.config_path,config)
    os.makedirs(config['labeled_save_dir'])
    copy2(os.path.join('./hand_label_context_tool', 'emb.npy'), os.path.join(config['labeled_save_dir'], 'emb.npy'))

if not os.path.exists(config['labeled_save_dir']):
    os.makedirs(config['labeled_save_dir'])

server_frozen = False  # Whether the server is ready for new inputs or not.
# Setting up tools.
start_time = time.time()

database = server_util.initialize_database(config['database_path'])
active_labeling_util = active_learning_online_util.ActiveLabelingUtilOnline(
    config['cnn_model_path'], config['source_path'], config['target_path'],
    config['vocab_path'], config['sent_len'], config['labeled_save_dir'])
end_time = time.time()
print('Finished database and active learning initialization. Took %.1f seconds' % (end_time - start_time))

httpd = BaseHTTPServer.HTTPServer((args.host, args.port), MyHandler)
print 'serving at', args.host, ':', args.port
threading.Timer(3600, asynchronous_retrain).start()
httpd.serve_forever()