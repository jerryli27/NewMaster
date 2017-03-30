# Unit tests
import trie_util

import unittest
from collections import defaultdict
import tempfile
import os
import re

class ReadCorpusUtilTest(unittest.TestCase):
    def test_join_entity_names(self):
        s = 'This is a test string where we test whether join entity names is working properly on the test string ' \
            'the phrase join entity name should not be joined together but the phrase the test should'
        s = s.lower()
        trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names', 'the test'])
        output = trie_util.join_entity_names(s, trie)
        expected_output = \
            'This is a test_string where we test whether join_entity_names is working properly on the_test_string ' \
            'the phrase join entity name should not be joined together but the phrase the_test should'
        expected_output = expected_output.lower()
        self.assertEqual(output, expected_output)

    def test_count_entity_names(self):
        s = 'á THIS, is a test_string where we test ! Whether join_entity_names is working properly on the test string?!'
        trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names'])
        expected_output = defaultdict(int, {'test': 2, 'join_entity_names': 1, 'test_string': 1})
        actual_output = defaultdict(int)
        trie_util.count_entity_names(s, trie, actual_output)
        self.assertDictEqual(actual_output, expected_output)

    def test_get_key_phrase_pairs_from_context_sanity_check(self):
        s_before = 'á THIS, is a test_string where we test! Whether'
        s_after = 'join_entity_names is working properly on the test string?!'
        s_before_list = s_before.split(' ')
        s_after_list = s_after.split(' ')
        trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names'])
        expected_output = ('test_string', 'join_entity_names')
        actual_output = trie_util.get_key_phrase_pairs_from_context(s_before_list, s_after_list, trie)
        self.assertTupleEqual(actual_output, expected_output)

    def test_get_key_phrase_pairs_from_context_no_match_first(self):
        s_before = 'á THIS, is a where we test! Whether'
        s_after = 'join_entity_names is working properly on the test string?!'
        s_before_list = s_before.split(' ')
        s_after_list = s_after.split(' ')
        trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names'])
        expected_output = (None, None)
        actual_output = trie_util.get_key_phrase_pairs_from_context(s_before_list, s_after_list, trie)
        self.assertTupleEqual(actual_output, expected_output)


    def test_get_key_phrase_pairs_from_context_no_match_second(self):
        s_before = 'á THIS, is a test_string where we test! Whether'
        s_after = 'join entity names is working properly on the string?!'
        s_before_list = s_before.split(' ')
        s_after_list = s_after.split(' ')
        trie = trie_util.Trie(['test', 'test string', 'test case', 'the test string', 'join entity names'])
        expected_output = (None, None)
        actual_output = trie_util.get_key_phrase_pairs_from_context(s_before_list, s_after_list, trie)
        self.assertTupleEqual(actual_output, expected_output)
if __name__ == '__main__':
    unittest.main()