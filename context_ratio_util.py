"""
This file contains functions that measure the ratio between the occurrence of the phrase "A" versus the phrase
"A such as". This serves as an indicator for whether the phrase "A" is a common category word.
"""

import argparse
from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from read_corpus_util import kCorpusDirectory, read_key_phrases_dict

def count_key_phrases_around_context_in_preprocessed_sentence(sen, key_phrase_set, context):
    # type: (str, Set[str]) -> List[int]
    # Because the sentence should be pre_processed, words, phrases and punctuations are separated by space.
    """
    TODO:change def
    :param sen: A preprocessed sentence that possibly contains one or more key phrases.
    :param key_phrase_set: A set of key phrases.
    :return: The sorted indices of the key phrases in the sentence.
    """
    num_words_in_context = len(context)
    words = sen.split(' ')
    len_sentence = len(words)
    key_phrase_count = Counter()
    key_phrase_count_before_context = Counter()
    key_phrase_count_after_context = Counter()
    key_phrase_count_around_context = Counter()
    for i, word in enumerate(words):
        if word in key_phrase_set:
            key_phrase_count[word] = key_phrase_count.get(word,0) + 1
            # Check if context occurs before key phrase.
            if i >= num_words_in_context:
                success = True
                for context_word_i, context_word in enumerate(context):
                    if words[i - num_words_in_context + context_word_i] != context_word:
                        success = False
                        break
                if success:
                    key_phrase_count_after_context[word] = key_phrase_count_after_context.get(word,0) + 1
                    key_phrase_count_around_context[word] = key_phrase_count_around_context.get(word,0) + 1

            # Check if context occurs after key phrase.
            if i < len_sentence - num_words_in_context:
                success = True
                for context_word_i, context_word in enumerate(context):
                    if words[i + 1 + context_word_i] != context_word:
                        success = False
                        break
                if success:
                    key_phrase_count_before_context[word] = key_phrase_count_before_context.get(word,0) + 1
                    key_phrase_count_around_context[word] = key_phrase_count_around_context.get(word,0) + 1

    return key_phrase_count, key_phrase_count_around_context, key_phrase_count_before_context, key_phrase_count_after_context

def count_key_phrase_around_context_in_corpora(text_paths, key_phrase_set, context,
                                                     max_len=128, max_num_extract=1000000):
    # type: (List[str], Set[str], List[str], int, int, int) -> List[Tuple[str,List[Tuple[int,int]]]]

    key_phrase_count = Counter()
    key_phrase_count_before_context = Counter()
    key_phrase_count_after_context = Counter()
    key_phrase_count_around_context = Counter()
    num_sentence_extracted = 0
    for text_path in text_paths:
        with open(text_path, 'r') as text_file:
            paragraphs = text_file.readlines()
            for sentences in paragraphs:
                splitted_sentences = sent_tokenize(sentences.strip('\n'))
                for sentence in splitted_sentences:
                    if len(sentence) < max_len:
                        current_key_phrase_count, current_key_phrase_count_around_context, \
                        current_key_phrase_count_before_context, current_key_phrase_count_after_context = \
                            count_key_phrases_around_context_in_preprocessed_sentence(sentence, key_phrase_set, context)
                        if len(current_key_phrase_count) != 0:
                            key_phrase_count = key_phrase_count + current_key_phrase_count
                            key_phrase_count_around_context = key_phrase_count_around_context + current_key_phrase_count_around_context
                            key_phrase_count_before_context = key_phrase_count_before_context + current_key_phrase_count_before_context
                            key_phrase_count_after_context = key_phrase_count_after_context + current_key_phrase_count_after_context
                            num_sentence_extracted += 1
                            if num_sentence_extracted % 100 == 0:
                                print("Extracted %d sentences." % num_sentence_extracted)
                            if num_sentence_extracted >= max_num_extract:
                                print("Number of sentences extracted is larger than the maximum threshold. Stopping.")
                                return key_phrase_count, key_phrase_count_around_context, key_phrase_count_before_context, key_phrase_count_after_context
    return key_phrase_count, key_phrase_count_around_context, key_phrase_count_before_context, key_phrase_count_after_context

def store_context_ratio(text_paths, output_path, context = ['such','as'], corpus_directory = kCorpusDirectory,
                        to_lower = False, max_num_extract = 1000000):
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
    if context is None:
        raise AssertionError('context must not be None.')
    key_phrases_dict, key_phrases_set= read_key_phrases_dict(corpus_directory, to_lower=to_lower)
    key_phrases_set = set(map(lambda phrase: phrase.replace(' ','_'),list(key_phrases_set)))
    if "_".join(context) in key_phrases_set:
        raise AssertionError('%s is in the key phrase set.' %("_".join(context)))

    key_phrase_count, key_phrase_count_around_context, \
    key_phrase_count_before_context, key_phrase_count_after_context = \
        count_key_phrase_around_context_in_corpora(text_paths, key_phrases_set, context, max_num_extract)
    print("Finished finding co-occurrences.")

    # Then write to file in the format `key phrase`[tab]`ratio`
    with open(output_path, "w") as f:
        context_str = ' '.join(context)
        f.write("Key phrase\tCount\tAround %s probability\tKey phrase before %s probability\tKey phrase after %s probability\n"
                %(context_str,context_str,context_str))
        for key_phrase, count in key_phrase_count.iteritems():
            f.write("%s\t%d\t%f\t%f\t%f\n" %(key_phrase, count,
                                             float(key_phrase_count_around_context[key_phrase]) / count,
                                             float(key_phrase_count_before_context[key_phrase]) / count,
                                             float(key_phrase_count_after_context[key_phrase]) / count))

def main(args):
    store_context_ratio(text_paths=args.text_paths, output_path=args.output_path, context=args.context.split(" "),
                        corpus_directory=args.corpus_directory, to_lower=args.to_lower, max_num_extract=args.max_num_extract)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_paths", required=True, help="Paths to the text corpora.", nargs='*')
    parser.add_argument("--output_path", required=True, help="The file name of the output file.")
    parser.add_argument("--context", default="such as",
                        help="The context around which key phrase appears. "
                             "The ratio will be calculated based on this context.")
    parser.add_argument("--corpus_directory", default=kCorpusDirectory, help="Paths to the the cs corpus directory. "
                                                                             "Used to load the list of key phrases.")
    parser.add_argument("--to_lower", dest="to_lower", action="store_true",
                        help="Convert all corpus and vocab to lower case.")
    parser.set_defaults(to_lower=False)
    parser.add_argument("--max_num_extract", type=int, default=100000,
                        help="Maximum number of sentences the program goes through before stopping.")

    a = parser.parse_args()
    main(a)


