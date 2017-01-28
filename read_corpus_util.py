# This file contains functions for reading the  allenai paper corpus.
# It contains a directory "allenai-papers-corpus" which has one paper per line with tab-separated fields:
#
# - paper id
# - field of study (either "Computer Science" or "Neuroscience" or "")
# - paper title
# - paper abstract text
# - paper body text
#
# Additionally, the directory contains two other directories with:
# 1. key phrases assigned to each paper in the corpus
# 2. Neuroscience entities assigned to neuroscience papers (include brain regions, cell types and organism)

import ast
import csv
import os
import re
import string

import progressbar

import trie_util
from trie_util import join_entity_names

kCorpusDirectory = "/home/xor/MasterData/"
kRegexAlphaNumOnly = re.compile('([^\sa-zA-Z]|_)+')
kRegexMultipleWhiteSpace = re.compile('\s\s+')
kRegexAsciiOnly = re.compile('[^\x00-\x7F]+')


def all_files_in_dir(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


# Given the directory to the corpus, tries to find the subfolder containing neural science entities, parse the files,
# and return one dictionary with key = paperid and value = Neuroscience entities assigned to neuroscience papers
# (include brain regions, cell types and organism). It also returns a facet dictionary with key = facet type and value
# = set of entity names in the facet. Lastly, it returns a set containing the entity names.
def read_ns_entities(corpus_directory, to_lower=True, ns_entities_folder = 'nsEntities'):
    ns_entities_directory = corpus_directory + '/' + ns_entities_folder + '/'
    files = all_files_in_dir(ns_entities_directory)
    paper_ns_entities_dict = {}  # key = paperid and value = Neuroscience entities stored as facets.
    facet_dict = {}  # key = facetType value = entity names.
    entity_names_set = set()
    num_papers = 0
    bar = progressbar.ProgressBar(max_value=len(files))

    paper_id_string = 'paperid' if to_lower else 'paperId'
    facet_type_string = 'facettype' if to_lower else 'facetType'

    for i_files, file_name in enumerate(files):
        with open(ns_entities_directory + file_name, 'r') as f:
            for line in f:
                num_papers += 1
                current_dict = ast.literal_eval(line.lower() if to_lower else line)
                paper_ns_entities_dict[current_dict[paper_id_string]] = current_dict['facets']
                for facet in current_dict['facets']:
                    if facet[facet_type_string] not in facet_dict:
                        facet_dict[facet[facet_type_string]] = set()
                    for facet_entity_names in facet['values']:
                        facet_dict[facet[facet_type_string]].add(facet_entity_names)
                        entity_names_set.add(facet_entity_names)
        bar.update(i_files)
    assert num_papers == len(paper_ns_entities_dict)
    print(
        "\nFinished reading ns entities. There are %d papers with %d ns entities in total."
        % (len(paper_ns_entities_dict), len(entity_names_set)))
    return paper_ns_entities_dict, facet_dict, entity_names_set


# This function tries to find the key phrases directory under the corpus directory, open the files, and output a dict
# with key = paper id and value = key phrases. It also outputs a set of key phrases in all the papers.
def read_key_phrases(corpus_directory, to_lower=True, key_phrases_folder = 'keyphrases'):
    key_phrases_directory = corpus_directory + '/' + key_phrases_folder + '/'
    files = all_files_in_dir(key_phrases_directory)
    paper_key_phrases_dict = {}  # key = paperid and value = key phrases for that paper.
    key_phrases_set = set()
    num_papers = 0
    bar = progressbar.ProgressBar(max_value=len(files))
    for i_files, file_name in enumerate(files):
        with open(key_phrases_directory + file_name, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            for line in csv_reader:
                num_papers += 1
                paper_key_phrases_dict[line[0]] = line[1:]
                for key_phrase in line[1:]:
                    if to_lower:
                        key_phrases_set.add(key_phrase.lower())
                    else:
                        key_phrases_set.add(key_phrase)
        bar.update(i_files)
    assert num_papers == len(paper_key_phrases_dict)
    print("\nFinished reading ns entities. There are %d papers and %d unique key phrases in total."
          % (len(paper_key_phrases_dict), len(key_phrases_set)))
    return paper_key_phrases_dict, key_phrases_set


# This function finds the papers folder under corpus directory, and returns a string containing only preprocessed
# paper body text. Preprocessing includes throwing out papers not belonging to the given field, keeping only
# alphanumerical characters, turning them into lower case, and replace words with entities (words joined together by _)
def read_papers_corpus(corpus_directory, field_of_study=None, start_index=None, end_index=None, trie=None,
                       papers_corpus_folder = 'allenai-papers-corpus', to_lower=False, no_punctuations=False,
                       no_special_char=False, no_preprocess = False):
    ns_entities_directory = corpus_directory + '/' + papers_corpus_folder + '/'
    files = all_files_in_dir(ns_entities_directory)
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(files)
    num_papers = 0
    bar = progressbar.ProgressBar(max_value=(end_index - start_index))
    corpus = []
    for i_files, file_name in enumerate(files[start_index:end_index]):
        with open(ns_entities_directory + file_name, 'r') as f:
            # Can't use csv_reader because line is too long.
            for line in f:
                # Each row contains the following:
                # - paper id
                # - field of study (either "Computer Science" or "Neuroscience" or "")
                # - paper title
                # - paper abstract text
                # - paper body text
                line = line.split('\t')
                if field_of_study == None or field_of_study == line[1]:
                    num_papers += 1
                    if no_preprocess:
                        corpus.append(line[3])
                        corpus.append(line[4])
                    else:
                        corpus.append(preprocess_corpus(line[3], trie, to_lower, no_punctuations, no_special_char))
                        corpus.append(preprocess_corpus(line[4], trie, to_lower, no_punctuations, no_special_char))
        bar.update(i_files)
    corpus = " ".join(corpus)  # Get rid of the spaces in the beginning and at the end, if any.
    print("\nFinished reading all papers. There are %d papers in total."
          % (num_papers))
    return corpus


# For now, this function only preprocess the string corpus by turning it into lower cases and represent entity names as
# one word by joining multiple words together using underscore.
# It also get rid of all non-alphanumerical characters including punctuations.
def preprocess_corpus(s, trie, to_lower=False, no_punctuations=False, no_special_char=False):
    # Getting rid of all non-alphanumerical characters except spaces and remove multiple white space.
    if to_lower:
        if no_punctuations:
            if no_special_char:
                return join_entity_names(kRegexMultipleWhiteSpace.sub(' ', kRegexAlphaNumOnly.sub('', s.lower())).strip(' '), trie)
            else:
                return join_entity_names(kRegexMultipleWhiteSpace.sub(' ', ''.join(c for c in s.lower() if c not in string.punctuation)).strip(' '), trie)
        else:
            if no_special_char:
                return join_entity_names(kRegexMultipleWhiteSpace.sub(' ', kRegexAsciiOnly.sub('', s.lower())).strip(' '), trie)
            else:
                return join_entity_names(kRegexMultipleWhiteSpace.sub(' ', s.lower()).strip(' '), trie)
    else:
        if no_punctuations:
            if no_special_char:
                return join_entity_names(kRegexMultipleWhiteSpace.sub(' ', kRegexAlphaNumOnly.sub('', s)).strip(' '), trie)
            else:
                return join_entity_names(kRegexMultipleWhiteSpace.sub(' ', ''.join(c for c in s if c not in string.punctuation)).strip(' '), trie)
        else:
            if no_special_char:
                return join_entity_names(kRegexMultipleWhiteSpace.sub(' ', kRegexAsciiOnly.sub('', s)).strip(' '), trie)
            else:
                return join_entity_names(kRegexMultipleWhiteSpace.sub(' ', s).strip(' '), trie)



# This function tries to find the key phrases dictionary file under the corpus directory, open the files, and output a
# dict with key = key phrases and value = number of occurrence. It also outputs a set of key phrases in all the papers.
def read_key_phrases_dict(corpus_directory, to_lower=False, key_phrases_folder = '.', count_threshold = 5):
    key_phrases_directory = corpus_directory + '/' + key_phrases_folder + '/keyphrases.dict'
    key_phrases_dict = {}
    key_phrases_set = set()

    with open(key_phrases_directory, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for line in csv_reader:
            if len(line) == 2:
                current_count = int(line[0])
                if current_count < count_threshold:
                    break
                if to_lower:
                    key_phrases_set.add(line[1].lower())
                    key_phrases_dict[line[1].lower()] = current_count
                else:
                    key_phrases_set.add(line[1])
                    key_phrases_dict[line[1]] = current_count
    print("\nFinished reading key phrase dict. There are %d unique key phrases in total that has count over %d."
          % (len(key_phrases_set), count_threshold))
    return key_phrases_dict, key_phrases_set
