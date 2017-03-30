# This trie class comes from leetcode forum.

import collections
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize

# 
class TrieNode(object):
    def __init__(self):
        self.is_phrase = False
        self.children = collections.defaultdict(TrieNode)
    # Returns None if the node does not have such a child with name = word. Return that child if it has it.
    def has_child(self, word):
        if word not in self.children:
            return None
        else:
            return self.children[word]
class Trie(object):
    def __init__(self, init_list = None):
        self.root = TrieNode()
        for item in init_list:
            self.insert(item)

    def insert(self, phrase):
        node = self.root
        for word in phrase.split(' '):
            node = node.children[word]
        node.is_phrase = True

    def search(self, phrase, is_phrase=True):
        node = self.root
        for word in phrase.split(' '):
            if word not in node.children:
                return False
            node = node.children[word]
        return (node.is_phrase if is_phrase else True, node)

    def starts_with(self, prefix):
        return self.search(prefix, False)


# Given a string with words separated by space and a trie structure containing entity names, do maximal match and
# join space-separated entity names with underscore.
def join_entity_names(s, trie):
    if trie is None:
        return s
    words = word_tokenize(s)
    words_merged = []
    len_words = len(words)
    i = 0
    while i < len_words:
        num_words = 0
        node = trie.root
        last_success = 0
        while i + num_words < len_words:
            next_node = node.has_child(words[i + num_words])
            if next_node is None:
                break
            else:
                node =next_node
                num_words += 1
                if node.is_phrase:
                    last_success = num_words
        # After the while loop, num_words-1 will be the number of word starting from index i that appeared in the trie.
        # We merge those words into one.
        last_success = max(1, last_success)
        words_merged.append('_'.join(words[i:i + last_success]))
        i += last_success
    return ' '.join(words_merged)



# Given a string with words separated by space and a trie structure containing entity names, do maximal match and
# modify the count_dict default dictionary containing key = underscore-separated entity names and value = count.
def count_entity_names(s, trie, count_dict):
    if trie is None:
        return s
    words = s.split(' ')
    # words = word_tokenize(s)
    words_merged = []
    len_words = len(words)
    i = 0
    while i < len_words:
        next_node = trie.root.has_child(words[i])
        if '_' in words[i] or (next_node is not None and next_node.is_phrase):
            count_dict[words[i]] += 1
        i += 1



# Given a string with words separated by space and a trie structure containing entity names, do maximal match and
# modify the count_dict default dictionary containing key = underscore-separated entity names and value = count.
def get_key_phrase_pairs_from_context(s_list_before, s_list_after, trie, ignore_same = True):
    ret = (None, None)
    if trie is None:
        return ret

    # First check the s_list_before starting from the last element. Stop when we see a key phrase.
    for i in range(len(s_list_before) - 1, -1, -1):
        s_before = s_list_before[i]
        s_before_node = trie.root.has_child(s_before)
        if '_' in s_before or (s_before_node is not None and s_before_node.is_phrase):
            ret = (s_before, None)
            break
    # If there is no key phrase in s_list_before, return a pair of Nones.
    if ret[0] is None:
        return ret
    # Otherwise, check whether s_list_after contains key phrase as well.
    for i in range(len(s_list_after)):
        s_after = s_list_after[i]
        s_after_node = trie.root.has_child(s_after)
        if '_' in s_after or (s_after_node is not None and s_after_node.is_phrase):
            # if ignore_same is true and the first and second key phrase are the same, ignore them.
            if not ignore_same or ret[0] != s_after:
                return (ret[0], s_after)
    return (None, None)