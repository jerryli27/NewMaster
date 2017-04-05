# Unit tests
from context_ratio_util import *

import unittest
import tempfile
import os
import re

import trie_util

class ReadCorpusUtilTest(unittest.TestCase):
    def test_count_key_phrases_around_context_in_preprocessed_sentence(self):
        sen = "this is just a test example . i will give an example such as this sentence ."
        key_phrase_set = {"example", "sentence", "test"}
        context = ['such','as']
        actual_output = count_key_phrases_around_context_in_preprocessed_sentence(sen, key_phrase_set, context)
        key_phrase_count = Counter({"example": 2 , "sentence": 1, "test": 1})
        key_phrase_count_before_context = Counter({"example": 1})
        key_phrase_count_after_context = Counter()
        key_phrase_count_around_context = Counter({"example": 1})
        expected_output = (key_phrase_count, key_phrase_count_around_context,
                           key_phrase_count_before_context, key_phrase_count_after_context)
        self.assertTupleEqual(actual_output, expected_output)


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