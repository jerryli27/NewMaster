"""
This is the unit test file for active learning util.
"""

import shutil
import tempfile
import unittest

from active_learning_offline_util import *

class ReadCorpusUtilTest(unittest.TestCase):
    def test_save_additional_label_1(self):
        sent_len = 3
        dirpath = tempfile.mkdtemp()
        unlabeled_data = np.array([[1,2,3,4,5],[6,7,8,9,0],[10,11,12,13,14]])
        additional_label_index = 0
        additional_label_result = np.array([[0,0,1]])
        labeled_data = np.array([[-1,2,3,4,5],[-6,7,8,9,0],[-10,11,12,13,14]])
        labels = np.array([[1,0,0],[0,1,0],[1,0,0]])
        source_path = dirpath + '/source'
        target_path = dirpath + '/target'
        save_additional_label(unlabeled_data, additional_label_index, additional_label_result, labeled_data, labels,
                              source_path,  target_path)

        labeled_x, labeled_y = util.read_data_labeled_part(source_path,target_path,sent_len,shuffle=False)
        unlabeled_x = util.read_data_unlabeled_part(source_path,target_path,sent_len,shuffle=False)
        actual_data = np.concatenate((labeled_x, unlabeled_x), axis=0)
        actual_labels = labeled_y
        expected_data = np.array([[-1,2,3,4,5],[-6,7,8,9,0],[-10,11,12,13,14],[1,2,3,4,5],[6,7,8,9,0],[10,11,12,13,14]])
        expected_labels =np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1]])
        shutil.rmtree(dirpath)

        np.testing.assert_array_equal(actual_data,expected_data)
        np.testing.assert_array_equal(actual_labels,expected_labels)
    def test_save_additional_label_2(self):
        sent_len = 3
        dirpath = tempfile.mkdtemp()
        unlabeled_data = np.array([[1,2,3,4,5],[6,7,8,9,0],[10,11,12,13,14]])
        additional_label_index = 1
        additional_label_result = np.array([[0,0,1]])
        labeled_data = np.array([[-1,2,3,4,5],[-6,7,8,9,0],[-10,11,12,13,14]])
        labels = np.array([[1,0,0],[0,1,0],[1,0,0]])
        source_path = dirpath + '/source'
        target_path = dirpath + '/target'
        save_additional_label(unlabeled_data, additional_label_index, additional_label_result, labeled_data, labels,
                              source_path,  target_path)

        labeled_x, labeled_y = util.read_data_labeled_part(source_path,target_path,sent_len,shuffle=False)
        unlabeled_x = util.read_data_unlabeled_part(source_path,target_path,sent_len,shuffle=False)
        actual_data = np.concatenate((labeled_x, unlabeled_x), axis=0)
        actual_labels = labeled_y
        expected_data = np.array([[-1,2,3,4,5],[-6,7,8,9,0],[-10,11,12,13,14],[6,7,8,9,0],[1,2,3,4,5],[10,11,12,13,14]])
        expected_labels =np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1]])
        shutil.rmtree(dirpath)

        np.testing.assert_array_equal(actual_data,expected_data)
        np.testing.assert_array_equal(actual_labels,expected_labels)
    def test_save_additional_label_3(self):
        sent_len = 3
        dirpath = tempfile.mkdtemp()
        unlabeled_data = np.array([[1,2,3,4,5],[6,7,8,9,0],[10,11,12,13,14]])
        additional_label_index = [0,2]
        additional_label_result = np.array([[0,0,1],[1,0,0]])
        labeled_data = np.array([[-1,2,3,4,5],[-6,7,8,9,0],[-10,11,12,13,14]])
        labels = np.array([[1,0,0],[0,1,0],[1,0,0]])
        source_path = dirpath + '/source'
        target_path = dirpath + '/target'
        save_additional_label(unlabeled_data, additional_label_index, additional_label_result, labeled_data, labels,
                              source_path,  target_path)

        labeled_x, labeled_y = util.read_data_labeled_part(source_path,target_path,sent_len,shuffle=False)
        unlabeled_x = util.read_data_unlabeled_part(source_path,target_path,sent_len,shuffle=False)
        actual_data = np.concatenate((labeled_x, unlabeled_x), axis=0)
        actual_labels = labeled_y
        expected_data = np.array([[-1,2,3,4,5],[-6,7,8,9,0],[-10,11,12,13,14],[1,2,3,4,5],[10,11,12,13,14],[6,7,8,9,0]])
        expected_labels =np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1],[1,0,0]])
        shutil.rmtree(dirpath)

        np.testing.assert_array_equal(actual_data,expected_data)
        np.testing.assert_array_equal(actual_labels,expected_labels)



if __name__ == '__main__':
    unittest.main()
