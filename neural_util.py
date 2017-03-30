"""
This file contains functions for tensorflow neural networks in general.
"""
from operator import mul

import numpy as np
import tensorflow as tf
from typing import Union, Tuple, List, Dict

def get_tensor_num_elements(tensor):
    # type: (tf.Tensor) -> int
    tensor_shape = map(lambda i: i.value, tensor.get_shape())
    return reduce(mul, tensor_shape, 1)

def get_tensor_shape(tensor):
    # type: (tf.Tensor) -> List[int]
    return map(lambda i: i.value, tensor.get_shape())


def embedding_lookup_in_sentence(sentence_i, position_index_pair, word2vec_lookup, relative_distance_2_vec_lookup):
    """

    :param sentence_i: tensor representing the sentence as the indices of words in the vocab. Shape = (num_batch,
    max_sentence_length)
    :param position_index_pair: 1-d tensor representing the indices of the pair of entities to be classified. Shape =
    (num_batch, 2)
    :param word2vec_lookup: The matrix with each row representing the word vector of the corresponding word.
    :param relative_distance_2_vec_lookup: The matrix with each row representing the word vector for one relative
    postion.
    :return: The word embedding of the given sentence as a Tensor.
    """
    sentence_len = get_tensor_shape(sentence_i)[1]

    # The shape for word_vectors = (num_batch, max_sentence_length, word2vec_dim)
    word_vectors = tf.nn.embedding_lookup(word2vec_lookup, sentence_i)
    word_to_key_phrase_1_vectors = []
    word_to_key_phrase_2_vectors = []
    for word_i in range(sentence_len):
        # The relative position ranges from 0 to 2*sentence_len - 2, representing 2 * sentence_len - 1 possible
        # difference in the two indices.
        relative_position_index_pair = tf.sub(np.int64(word_i + sentence_len - 1), position_index_pair)
        relative_position_1_index_pair, relative_position_2_index_pair = tf.unpack(relative_position_index_pair,
                                                                                    axis=1)
        # The shape for word_2_key_phrase_vector = (num_batch, relative_pos_embedding_dim)
        word_to_key_phrase_1_vector = tf.nn.embedding_lookup(relative_distance_2_vec_lookup,
                                                                   relative_position_1_index_pair)
        word_to_key_phrase_2_vector = tf.nn.embedding_lookup(relative_distance_2_vec_lookup,
                                                                    relative_position_2_index_pair)
        word_to_key_phrase_1_vectors.append(word_to_key_phrase_1_vector)
        word_to_key_phrase_2_vectors.append(word_to_key_phrase_2_vector)

    # The shape for word_2_key_phrase_vectors = (num_batch, max_sentence_length, relative_pos_embedding_dim)
    word_to_key_phrase_1_vectors = tf.pack(word_to_key_phrase_1_vectors, axis=1)
    word_to_key_phrase_2_vectors = tf.pack(word_to_key_phrase_2_vectors, axis=1)

    sentence_embeddings = tf.concat(2, (word_vectors, word_to_key_phrase_1_vectors, word_to_key_phrase_2_vectors))

    return sentence_embeddings

# def hide_key_phrases(sentence_i, position_index_pair,):
#     """
#
#     :param sentence_i: tensor representing the sentence as the indices of words in the vocab. Shape = (num_batch,
#     max_sentence_length)
#     :param position_index_pair: 1-d tensor representing the indices of the pair of entities to be classified. Shape =
#     (num_batch, 2)
#     :return: The `sentence_i` with indices at the two positions specified by `position_index_pair` replaced by 0, which
#     is the UNK word.
#     """
#     # position_1_index, position_2_index = tf.unpack(position_index_pair, axis=1)'
#     tf.scatter_update(sentence_i, position_index_pair, tf.zeros((tf.shape(sentence_i)[0],2)))
#     # sentence_i[:,position_1_index] = tf.zeros((tf.shape(sentence_i)[0],1))
#     # sentence_i[:,position_2_index] = tf.zeros((tf.shape(sentence_i)[0],1))
#     return sentence_i