"""
This file contains functions for preprocessing raw data to turn them into forms ready to be used by the CNN.
"""

import os
import numpy as np
import tensorflow as tf
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from typing import Union, Tuple, List, Dict, Set
from codecs import open as codecs_open


import read_corpus_util
import trie_util
import util
from neural_util import *


# Special vocabulary symbols.
PAD_TOKEN = '<pad>' # pad symbol
UNK_TOKEN = '<unk>' # unknown word
BOS_TOKEN = '<bos>' # begin-of-sentence symbol
EOS_TOKEN = '<eos>' # end-of-sentence symbol
NUM_TOKEN = '<num>' # numbers
UNK_WORD_VECTOR = None


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(THIS_DIR, 'hand_label_context_tool')

# we always put them at the start.
_START_VOCAB = [PAD_TOKEN, UNK_TOKEN]
PAD_ID = 0
UNK_ID = 1

def clean_raw_corpus(save_dir, start=0, end= 5000, step=100, to_lower=False, no_punctuations=False,
                   no_special_char=True):
    if save_dir[-1] != '/':
        raise AssertionError('save_dir must end with a "/".')

    # This reads all phrases.
    paper_dict, facet_dict, entity_names_set = read_corpus_util.read_ns_entities(read_corpus_util.kCorpusDirectory, to_lower=False)
    paper_key_phrases_dict, key_phrases_set = read_corpus_util.read_key_phrases(read_corpus_util.kCorpusDirectory, to_lower=False)
    all_phrases = list(entity_names_set.union(key_phrases_set))
    trie = trie_util.Trie(all_phrases)

    for i in range(start,end,step):
        # Field of study is either "Neuroscience" or "Computer Science"
        with open(save_dir + 'cs_corpus_max_matching_only_' + str(i) + '.txt', 'w', encoding='utf8') as f:
            f.write(read_corpus_util.read_papers_corpus(
            read_corpus_util.kCorpusDirectory, field_of_study = "Computer Science", start_index = i,
            end_index = i + step, trie = trie, to_lower=to_lower, no_punctuations=no_punctuations,
                       no_special_char=no_special_char))

def clean_text_file(save_dir, text_dir,to_lower=False, no_punctuations=False, no_special_char=True):

    key_phrases_dict, key_phrases_set= read_corpus_util.read_key_phrases_dict(read_corpus_util.kCorpusDirectory, to_lower=to_lower)
    key_phrases_list = map(lambda phrase: phrase.replace(' ','_'),list(key_phrases_set))
    trie = trie_util.Trie(key_phrases_list)
    with open(text_dir, 'r') as fin, open(save_dir, 'w') as fout:
        # Can't use csv_reader because line is too long.
        for line in fin:
            fout.write(read_corpus_util.preprocess_corpus(line, trie, to_lower, no_punctuations, no_special_char))
            fout.write('\n')


def extract_key_phrase_co_occurrences(text_paths, key_phrase_set, max_len=128, max_relative_distance = 15,
                                      max_num_extract = 10000):
    # type: (List[str], Set[str], int, int) -> List[Tuple[str,List[Tuple[int,int]]]]
    ret = []
    num_extracted = 0
    for text_path in text_paths:
        with open(text_path, 'r') as text_file:
            paragraphs = text_file.readlines()
            for sentences in paragraphs:
                splitted_sentences = sent_tokenize(sentences.strip('\n'))
                for sentence in splitted_sentences:
                    if len(sentence) < max_len:
                        kp_indices = find_key_phrases_in_preprocessed_sentence(sentence, key_phrase_set)
                        i_pairs = index_pairs_within_range(kp_indices, max_relative_distance)
                        if len(i_pairs) != 0:
                            ret.append((sentence,i_pairs))
                            num_extracted += 1
                            if num_extracted % 100 == 0:
                                print("Extracted %d sentences." % num_extracted)
                            if num_extracted >= max_num_extract:
                                print("Number of sentences extracted is larger than the maximum threshold. Stopping.")
                                return ret
    return ret

def extract_key_phrase_co_occurrences_around_context(text_paths, key_phrase_set, context,
                                                    max_len=128, max_relative_distance = 15,
                                      max_num_extract = 1000000):
    # type: (List[str], Set[str], List[str], int, int, int) -> List[Tuple[str,List[Tuple[int,int]]]]
    ret = []
    num_extracted = 0
    for text_path in text_paths:
        with open(text_path, 'r') as text_file:
            paragraphs = text_file.readlines()
            for sentences in paragraphs:
                splitted_sentences = sent_tokenize(sentences.strip('\n'))
                for sentence in splitted_sentences:
                    if len(sentence) < max_len:
                        kp_pairs = find_key_phrases_in_preprocessed_sentence_around_context(sentence,
                                                                                              key_phrase_set, context)
                        if len(kp_pairs) != 0:
                            ret.append((sentence,kp_pairs))
                            num_extracted += len(kp_pairs)
                            if num_extracted % 100 == 0:
                                print("Extracted %d sentences." % num_extracted)
                            if num_extracted >= max_num_extract:
                                print("Number of sentences extracted is larger than the maximum threshold. Stopping.")
                                return ret
    return ret

def find_key_phrases_in_preprocessed_sentence(sen, key_phrase_set):
    # type: (str, Set[str]) -> List[int]
    # Because the sentence should be pre_processed, words, phrases and punctuations are separated by space.
    """

    :param sen: A preprocessed sentence that possibly contains one or more key phrases.
    :param key_phrase_set: A set of key phrases.
    :return: The sorted indices of the key phrases in the sentence.
    """
    words = sen.split(' ')
    ret = []
    same_phrases = set()
    for i, word in enumerate(words):
        if word in key_phrase_set and word not in same_phrases:
            ret.append(i)
            same_phrases.add(word)
    return sorted(ret)

def find_key_phrases_in_preprocessed_sentence_around_context(sen, key_phrase_set, context):
    # type: (str, Set[str]) -> List[int]
    # Because the sentence should be pre_processed, words, phrases and punctuations are separated by space.
    """

    :param sen: A preprocessed sentence that possibly contains one or more key phrases.
    :param key_phrase_set: A set of key phrases.
    :return: The sorted indices of the key phrases in the sentence.
    """
    words = sen.split(' ')
    ret = []


    for i, _ in enumerate(words):
        if i == 0 or i == len(words) - 1:
            continue
        success = True
        for word_i, word in enumerate(context):
            if words[i + word_i] != word:
                success = False
                break
        if success:
            # TODO: for now only look for the word right before and after the context.
            if words[i-1] in key_phrase_set and words[i+len(context)] in key_phrase_set and words[i-1] != words[i+len(context)]:
                ret.append([i-1,i+len(context)])
    return ret

def index_pairs_within_range(index_list, d):
    # type: (List[int], int) -> List[Tuple[int,int]]
    """

    :param index_list: List of indices.
    :param d: The largest difference between two indices for them to be included in the return.
    :return: A list of indices that has a relative distance of d between them. One pair will only appear once
    regardless of their relative order
    """
    l = len(index_list)
    i1 = 0
    i2_end = 0
    ret = []
    while i1 < l:
        # Move i2_end until the element in list 2 is the smallest one that is greater than the element in list 1 + d.
        while i2_end < l and index_list[i1] + d >= index_list[i2_end]:
            i2_end += 1
        for i2 in range(i1+1, i2_end):
            ret.append((index_list[i1],index_list[i2]))
        i1 += 1
    return ret

def split_and_pad_sentence(sentence, max_sentence_length, bos=True, eos=True):
    words = ([BOS_TOKEN] if bos else []) + sentence.strip().split() + ([EOS_TOKEN] if eos else [])
    if len(words) > max_sentence_length:
        print('Max sentence length exceeded. Returning None.')
        return None

    words = words + [PAD_TOKEN for _ in range(max_sentence_length - len(words))]
    assert len(words) == max_sentence_length
    return words

def create_vocabulary(vocabulary_path, sentences, max_vocabulary_size=40000):
    """Create vocabulary file (if it does not exist yet) from data file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    # if not os.path.exists(vocabulary_path):
    print("Creating vocabulary %s." % (vocabulary_path))
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    for start_vocab in _START_VOCAB[::-1]:
        if start_vocab in vocab_list:
            vocab_list.insert(0, vocab_list.pop(vocab_list.index(start_vocab)))
        else:
            vocab_list.insert(0,start_vocab)
    if len(vocab_list) > max_vocabulary_size:
        print("  %d words found. Truncate to %d." % (len(vocab_list), max_vocabulary_size))
        vocab_list = vocab_list[:max_vocabulary_size]

    with codecs_open(vocabulary_path, "wb", encoding="utf-8") as vocab_file:
        for w in vocab_list:
            vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with codecs_open(vocabulary_path, "rb", encoding="utf-8") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentences_to_indices(sentences, vocab):
    ret = []
    for sentence in sentences:
        sentence_indices = []
        for word in sentence:
            sentence_indices.append(vocab.get(word, 0))  # 0 is the index for UNK.
        ret.append(sentence_indices)
    return ret


def indices_to_sentences(indices, rev_vocab, ignore_pad = True):
    rec_vocab_len = len(rev_vocab)
    ret = []
    for sentence_indices in indices:
        sentence = []
        for index in sentence_indices:
            if ignore_pad and index == 0:
                break
            sentence.append(rev_vocab[index] if index < rec_vocab_len else 'INDEX_OUT_OF_RANGE')  # 0 is the index for UNK.
        ret.append(sentence)
    return ret


def preprocess_from_text_to_unlabeled_data(text_paths, vocab_save_path, unlabeled_data_save_path, context = ['such','as'],
                                           corpus_directory = read_corpus_util.kCorpusDirectory, to_lower = False,
                                           max_num_extract = 1000000):
    # type: (List[str], str, str, Union[List[str],None], str, bool, int) -> None
    """
    Read from "text_paths", find all candidate sentences where two key phrases appears close enough to each
    other, create a vocab from the sentences, and record the result with the last two dimension as the indices of the
    two key phrases and the rest as the indices of words in the vocab.
    :param text_paths:
    :param vocab_save_path:
    :param unlabeled_data_save_path:
    :param corpus_directory:
    """
    # paper_dict, facet_dict, entity_names_set = read_corpus_util.read_ns_entities(corpus_directory, to_lower=False)
    # paper_key_phrases_dict, key_phrases_set = read_corpus_util.read_key_phrases(corpus_directory, to_lower=False)
    key_phrases_dict, key_phrases_set= read_corpus_util.read_key_phrases_dict(corpus_directory, to_lower=to_lower)
    key_phrases_set = set(map(lambda phrase: phrase.replace(' ','_'),list(key_phrases_set)))
    if 'such_as' in key_phrases_set:
        raise AssertionError('such as is in the key phrase set.')

    if context is None:
        sentence_key_phrase_pairs_list = extract_key_phrase_co_occurrences(text_paths,key_phrases_set, max_num_extract)
    else:
        sentence_key_phrase_pairs_list = extract_key_phrase_co_occurrences_around_context(text_paths,key_phrases_set, context, max_num_extract)
    print("Finished finding co-occurrences.")

    kMaxSentenceLength = 128

    preprocessed_sentences = []
    for sentence, key_phrase_indices in sentence_key_phrase_pairs_list:
        preprocessed_sentences.append(split_and_pad_sentence(sentence,kMaxSentenceLength))
    print("Finished splitting and padding sentences. Number of sentences: %d" %(len(preprocessed_sentences)))


    create_vocabulary(vocab_save_path, preprocessed_sentences)
    vocab, rev_vocab = initialize_vocabulary(vocab_save_path)
    print("Finished creating vocab.")

    sentence_indices_list = sentences_to_indices(preprocessed_sentences,vocab)
    print("Finished converting sentences to indices.")

    sentence_concat_key_phrase = []
    for sentence_i, sentence_indices in enumerate(sentence_indices_list):
        # split_and_pad_sentence could return None.
        if sentence_indices is not None:
            for key_phrase_index_pair in sentence_key_phrase_pairs_list[sentence_i][1]:
                # The plus one is for padding...
                sentence_concat_key_phrase.append(np.concatenate((np.array(sentence_indices, dtype=np.int),
                                                                  np.array(key_phrase_index_pair, dtype=np.int) + 1),
                                                                 axis=0))

    sentence_concat_key_phrase = np.array(sentence_concat_key_phrase, dtype=np.int)
    assert sentence_concat_key_phrase.shape[1] == kMaxSentenceLength + 2
    np.savetxt(unlabeled_data_save_path, sentence_concat_key_phrase, delimiter=' ', fmt='%d')


def combine_preprocessed(file_dir_tuples, vocab_save_path, unlabeled_data_save_path, labels_save_path):
    # type: (List[Tuple[str,str,str]],str,str) -> None
    preprocessed_sentences = []
    key_phrase_indices = []
    is_labeled = []
    labels = []

    for vocab_path, unlabeled_data_path, labels_path in file_dir_tuples:

        unlabeled_data = np.loadtxt(unlabeled_data_path,dtype=np.int,delimiter=' ')
        sentence_indices = unlabeled_data[:, :-2]
        _, rev_vocab = initialize_vocabulary(vocab_path)
        sentences = indices_to_sentences(sentence_indices, rev_vocab, ignore_pad=False)
        key_phrase_indices += unlabeled_data[:, -2:].tolist()
        preprocessed_sentences += sentences

        current_labels = util.read_labels_file(labels_path)
        is_labeled += [True] * len(current_labels) + [False] * (len(sentences) - len(current_labels))
        labels += current_labels

    assert len(preprocessed_sentences) == len(key_phrase_indices) and len(preprocessed_sentences) == len(is_labeled)

    # Now move all sentences with labels towards the head.
    preprocessed_sentences_reordered = []
    key_phrase_indices_reordered = []
    for sentence_i, sentence in enumerate(preprocessed_sentences):
        if is_labeled[sentence_i]:
            preprocessed_sentences_reordered.append(sentence)
            key_phrase_indices_reordered.append(key_phrase_indices[sentence_i])
    for sentence_i, sentence in enumerate(preprocessed_sentences):
        if not is_labeled[sentence_i]:
            preprocessed_sentences_reordered.append(sentence)
            key_phrase_indices_reordered.append(key_phrase_indices[sentence_i])
    preprocessed_sentences = preprocessed_sentences_reordered
    key_phrase_indices = key_phrase_indices_reordered

    create_vocabulary(vocab_save_path, preprocessed_sentences)
    vocab, rev_vocab = initialize_vocabulary(vocab_save_path)
    print("Finished creating vocab.")

    sentence_indices_list = sentences_to_indices(preprocessed_sentences,vocab)
    print("Finished converting sentences to indices.")

    sentence_concat_key_phrase = []
    for sentence_i, sentence_indices in enumerate(sentence_indices_list):
        # split_and_pad_sentence could return None.
        if sentence_indices is not None:
            # The plus one is for padding...
            sentence_concat_key_phrase.append(np.concatenate((np.array(sentence_indices, dtype=np.int),
                                                              np.array(key_phrase_indices[sentence_i], dtype=np.int)),
                                                             axis=0))

    sentence_concat_key_phrase = np.array(sentence_concat_key_phrase, dtype=np.int)
    np.savetxt(unlabeled_data_save_path, sentence_concat_key_phrase, delimiter=' ', fmt='%d')

    # Lastly save the labels
    util.save_labels_file(labels,labels_save_path)


if __name__=="__main__":
    # preprocess_from_text_to_unlabeled_data(['/home/xor/MasterData/sample/tiny.txt'], 'data/test_vocab',
    #                                        'data/test_unlabeled_data.csv')
    # clean_text_file('/home/xor/MasterData/cs_corpus_test.txt',
    #                 '/home/xor/MasterData/cs_corpus_en_and_kp_replaced_punc_included0.txt', to_lower=True,
    #                 no_punctuations=False,no_special_char=True)
    # preprocess_from_text_to_unlabeled_data(['/home/xor/MasterData/cs_corpus_test.txt'],
    #                                        'data/test_cs_vocab_such_as',
    #                                        'data/test_cs_unlabeled_data_such_as.txt', to_lower=True)

    preprocess_from_text_to_unlabeled_data(['/home/xor/MasterData/cs_corpus_test.txt'],
                                           'data/test_cs_vocab_random_large',
                                           'data/test_cs_unlabeled_data_random_large.txt',context=None,to_lower=True, max_num_extract=1000000)

    # # pretrained embeddings
    # embedding_path = '/media/xor/D/google300/GoogleNews-vectors-negative300.bin'
    # vocab_path = os.path.join(DATA_DIR,'test_cs_vocab_combined')
    # if os.path.exists(embedding_path):
    #     word2id, _ = initialize_vocabulary(vocab_path)
    #     embedding = util.prepare_pretrained_embedding(embedding_path, word2id)
    #     np.save(os.path.join(DATA_DIR,'emb.npy'), embedding)
    # else:
    #     print "Pretrained embeddings file %s not found." % embedding_path

    # combine_preprocessed([('hand_label_context_tool/test_cs_vocab_such_as',
    #                        'hand_label_context_tool/test_cs_unlabeled_data_such_as.txt',
    #                        'hand_label_context_tool/test_cs_labels_such_as.txt'),
    #                       ('hand_label_context_tool/test_cs_vocab_threshold_5',
    #                        'hand_label_context_tool/test_cs_unlabeled_data_threshold_5.txt',
    #                        'hand_label_context_tool/test_cs_labels_threshold_5.txt')
    #                       ],'data/test_cs_vocab_combined','data/test_cs_unlabeled_data_combined.txt',
    #                      'data/test_cs_labels_combined.txt')
    pass