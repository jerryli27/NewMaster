from infer_labels_util import *

import unittest
import tempfile
import os
import re
import shutil

import trie_util
from preprocessing_util import create_vocabulary

class InferLabelsUtilTest(unittest.TestCase):
    def test_infer_from_labeled(self):
        target_path = "/home/jerryli27/PycharmProjects/NewMaster/sanity_check_co_training/1492631467/test_cs_labels_combined_round_2.txt"
        source_path = "/home/jerryli27/PycharmProjects/NewMaster/sanity_check_co_training/1492631467/test_cs_unlabeled_data_combined_round_2.txt"
        vocab_path = "/home/jerryli27/PycharmProjects/NewMaster/inferred_dataset/test_cs_vocab_combined"
        infer_from_labeled(source_path, target_path, 128, vocab_path, do_save=True, save_target_path=target_path,
                           save_source_path=source_path)

        infer_from_labeled(source_path, target_path, 128, vocab_path, do_save=True, save_target_path=target_path,
                           save_source_path=source_path)

        infer_from_labeled(source_path, target_path, 128, vocab_path, do_save=True, save_target_path=target_path,
                           save_source_path=source_path)

        infer_from_labeled(source_path, target_path, 128, vocab_path, do_save=True, save_target_path=target_path,
                           save_source_path=source_path)

        # dirpath = tempfile.mkdtemp()
        # try:
        #     source_path = os.path.join(dirpath, "source.tsv")
        #     target_path = os.path.join(dirpath, "target.tsv")
        #     vocab_path = os.path.join(dirpath, "vocab")
        #
        #     # Write vocab:
        #     sentences = ["Sentence one".split(" "), "Sentence two".split(" "), ]
        #     create_vocabulary(vocab_path,sentences)
        #
        #     with open(source_path, "w") as f:
        #         f.write("1 2 0 0 0 0 0 1\n")
        #         f.write("2 3 0 0 0 0 0 1\n")
        #         f.write("3 4 0 0 0 0 0 1\n")
        #         f.write("1 2 0 0 0 0 0 1\n")
        #         f.write("1 2 0 0 0 0 0 1\n")
        #         f.write("1 2 0 0 0 0 0 1\n")
        #         f.write("1 2 0 0 0 0 0 1\n")
        #         f.write("1 2 0 0 0 0 0 1\n")
        #         f.write("2 3 0 0 0 0 0 1\n")
        #         f.write("4 5 0 0 0 0 0 1\n")
        #         f.write("2 1 0 0 0 0 0 1\n")
        #         f.write("3 2 0 0 0 0 0 1\n")
        #
        #     with open(target_path, "w") as f:
        #         f.write("0 0 1\n")
        #         f.write("1 0 0\n")
        #
        #
        #     additional_label_index, additional_label_result = infer_from_labeled(source_path,target_path,6,vocab_path,
        #                                                                          do_save=True,
        #                                                                          save_source_path=source_path,
        #                                                                          save_target_path=target_path)
        #     expected_additional_label_index = [1,2,3,4,5,6,8,9]
        #     expected_additional_label_result = [[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[1,0,0],[0,0,1],[0,1,0]]
        #
        #     self.assertListEqual(additional_label_index, expected_additional_label_index)
        #     for i in range(len(expected_additional_label_result)):
        #         self.assertListEqual(additional_label_result[i].tolist(), expected_additional_label_result[i])
        #
        #
        #     with open(source_path, "r") as f:
        #         actual_saved_source = f.readlines()
        #     from codecs import open as codecs_open
        #     with codecs_open(target_path, mode="r", encoding="utf-8-sig") as f:
        #         actual_saved_target = f.readlines()
        #
        #     expected_saved_source = ["1 2 0 0 0 0 0 1\n",
        #                                 "2 3 0 0 0 0 0 1\n",
        #                                 "1 2 0 0 0 0 0 1\n",
        #                                 "1 2 0 0 0 0 0 1\n",
        #                                 "1 2 0 0 0 0 0 1\n",
        #                                 "1 2 0 0 0 0 0 1\n",
        #                                 "1 2 0 0 0 0 0 1\n",
        #                                 "2 3 0 0 0 0 0 1\n",
        #                                 "2 1 0 0 0 0 0 1\n",
        #                                 "3 2 0 0 0 0 0 1\n",
        #                                 "3 4 0 0 0 0 0 1\n",
        #                                 "4 5 0 0 0 0 0 1\n",]
        #     expected_saved_target = ["0 0 1\n","1 0 0\n","0 0 1\n","0 0 1\n","0 0 1\n","0 0 1\n","0 0 1\n","1 0 0\n","0 0 1\n","0 1 0\n",]
        #     self.assertListEqual(actual_saved_source, expected_saved_source)
        #     self.assertListEqual(actual_saved_target, expected_saved_target)
        # finally:
        #     shutil.rmtree(dirpath)

    # def test_read_papers_corpus(self):
    #     # First create temporary directories and files for testing
    #     with tempfile.TemporaryDirectory() as tmp_dir_name:
    #         print('created temporary directory', tmp_dir_name)
    #         os.makedirs(tmp_dir_name + '/allenai-papers-corpus/')
    #         with open(tmp_dir_name + '/allenai-papers-corpus/test', 'w') as f:
    #             f.write(')
    #
    #         corpus = read_corpus_util.read_papers_corpus(tmp_dir_name, to_lower=False, no_punctuations=True,
    #                                                      no_special_char=False)
    #         expected_corpus=''
    #         self.assertEqual(corpus, expected_corpus)


if __name__ == '__main__':
    unittest.main()