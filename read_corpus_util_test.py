# Unit tests
import read_corpus_util

import unittest
import tempfile
import os
import re

import trie_util

class ReadCorpusUtilTest(unittest.TestCase):
    def test_preprocess_corpus(self):
        s = 'á THIS, is a test string, where we test ! Whether join entity names is working properly on the test string?!'
        trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names'])
        output = read_corpus_util.preprocess_corpus(s, trie, to_lower=True, no_punctuations=True, no_special_char=True)
        expected_output = \
            'this is a test_string where we test whether join_entity_names is working properly on the_test_string'
        expected_output = expected_output.lower()
        self.assertEqual(output, expected_output)

        def test_preprocess_corpus_no_lower(self):
            s = 'á THIS, is a test string, where we test ! Whether join entity names is working properly on the test string?!'
            trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names'])
            output = read_corpus_util.preprocess_corpus(s, trie, to_lower=False, no_punctuations=True,
                                                        no_special_char=True)
            expected_output = \
                'THIS is a test_string where we test Whether join_entity_names is working properly on the_test_string'
            self.assertEqual(output, expected_output)

    def test_preprocess_corpus_no_lower_with_punc(self):
        s = 'á THIS, is a test string, where we test ! Whether join entity names is working properly on the test string?!'
        trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names'])
        output = read_corpus_util.preprocess_corpus(s, trie, to_lower=False, no_punctuations=False, no_special_char=True)
        # Notice that when there is punctuation, we cannot connect 'test string,' to 'test_string,'.
        expected_output = \
            'THIS, is a test string, where we test ! Whether join_entity_names is working properly on the test string?!'
        self.assertEqual(output, expected_output)

        def test_preprocess_corpus_no_lower_with_punc_with_special(self):
            s = 'á THIS, is a test string, where we test ! Whether join entity names is working properly on the test string?!'
            trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names'])
            output = read_corpus_util.preprocess_corpus(s, trie, to_lower=False, no_punctuations=False,
                                                        no_special_char=False)
            expected_output = \
                'á THIS, is a test string, where we test ! Whether join_entity_names is working properly on the test string?!'
            self.assertEqual(output, expected_output)

    def test_preprocess_corpus_no_lower_no_punc_with_special(self):
        s = 'á THIS, is a test string, where we test ! Whether join entity names is working properly on the test string?!'
        trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names'])
        output = read_corpus_util.preprocess_corpus(s, trie, to_lower=False, no_punctuations=True, no_special_char=False)
        expected_output = \
            'á THIS is a test_string where we test Whether join_entity_names is working properly on the_test_string'
        self.assertEqual(output, expected_output)

    def test_read_ns_entities(self):
        # First create temporary directories and files for testing
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print('created temporary directory', tmp_dir_name)
            os.makedirs(tmp_dir_name + '/nsEntities/')
            with open(tmp_dir_name + '/nsEntities/test', 'w') as f:
                f.write('{"paperId":"1","facets":[{"facetType":"organism","values":["Human"],'
                        '"hierarchicalValues":["organism/Human"]}]}')

            paper_dict, facet_dict, entity_names_set = read_corpus_util.read_ns_entities(tmp_dir_name)
            expected_paper_dict = {"1":[{"facettype":"organism","values":["human"],
                                         "hierarchicalvalues":["organism/human"]}]}
            expected_facet_dict = {"organism":{"human"}}
            expected_entity_names_set = {"human"}
            self.assertDictEqual(paper_dict, expected_paper_dict)
            self.assertDictEqual(facet_dict, expected_facet_dict)
            self.assertSetEqual(entity_names_set, expected_entity_names_set)

    def test_read_papers_corpus(self):
        # First create temporary directories and files for testing
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print('created temporary directory', tmp_dir_name)
            os.makedirs(tmp_dir_name + '/allenai-papers-corpus/')
            with open(tmp_dir_name + '/allenai-papers-corpus/test', 'w') as f:
                f.write('1\tComputer Science\ttitle\tAbstráct\tbody!!! ???')

            corpus = read_corpus_util.read_papers_corpus(tmp_dir_name, to_lower=False, no_punctuations=True,
                                                         no_special_char=False)
            expected_corpus='Abstráct body'
            self.assertEqual(corpus, expected_corpus)

    # TODO: write test for read_key_phrases

if __name__ == '__main__':
    unittest.main()