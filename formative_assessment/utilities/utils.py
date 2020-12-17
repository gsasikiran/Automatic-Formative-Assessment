""" Creates various helper function for preprocessing and processing.

PreProcess class consists various functions such as tokenization, normalization etc.,
Utilities consists of the helper functions for processing various functions, that do not
go into any other classes.
"""
import pickle
import re
import string
from typing import List

import neuralcoref
import numpy as np
import pytextrank
import spacy
from allennlp.predictors import Predictor
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import WordNetLemmatizer
from scipy.spatial.distance import cosine
from spacy.matcher import Matcher
from nltk.corpus import wordnet

from formative_assessment.utilities.preprocessing import PreProcess
from formative_assessment.utilities.embed import Embedding

__author__ = "Sasi Kiran Gaddipati"
__credits__ = []
__license__ = ""
__version__ = ""
__last_modified__ = "14.12.2020"
__status__ = "Development"


class Utilities:
    def __init__(self):
        self.preprocess = PreProcess()

        self.nlp = spacy.load("en_core_web_lg")
        neuralcoref.add_to_pipe(self.nlp)
        tr = pytextrank.TextRank()
        self.nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)

        self.predictor = Predictor.from_path("weights/openie-model.2020.03.26.tar.gz")

        self.chunk_tagger = SequenceTagger.load('chunk')

    @staticmethod
    def _get_embed_list(tokens):
        with open("dataset/embeddings/phrases_use_stu_answers.pickle", "rb") as handle:
            phrases_embed = pickle.load(handle)

        embed_list = []
        for chunk in tokens:
            embed_list.append(phrases_embed[chunk])

        return embed_list

    @staticmethod
    def get_cosine_similarity(array_1, array_2):
        return 1 - cosine(array_1, array_2)

    def cosine_similarity_matrix(self, array_1, array_2):
        """
            Creates a matrix with similarity values from USE embeddings for each token in text_a to each word in text_b
        :param array_1: List[np.array]

        :param array_2: List[np.array]

        :return: np array
            Returns the matrix with similarity values
        """

        matrix = np.zeros((len(array_2), len(array_1)))

        for i in range(0, len(array_2)):
            for j in range(0, len(array_1)):
                matrix[i][j] = float(1 - self.get_cosine_similarity(array_2[i], array_1[j]))
        return matrix

    @staticmethod
    def get_frequency(desired_words, total_tokens):
        """
            Counts the occurrence of words in the total text
        :param desired_words: List[str]
            Required words to count
        :param total_tokens: List[str]
            All the tokens in the text with as many counts as occurred
        :return: dict
            Returns the dictionary of desired words as keys with corresponding count as values
        """
        word_freq = {}

        for word in desired_words:
            count = 0
            for answer in total_tokens:
                if word in answer:
                    count += 1

            word_freq[word] = count

        return word_freq

    def corefer_resolution(self, text):
        """
            Resolves the coreference resolution. Change the occurence of coreference to the actual word
        :param text: str
            Raw text that consists of
        :return:
        """

        doc = self.nlp(text.lower())
        return doc._.coref_resolved

    def extract_phrases(self, text: str):
        """
            Extracts the phrases of the text extracted from Flair package
        :param text: string
        :return: List[str]
            Returns the extracted list of phrases from the input text
        """

        sentence = Sentence(text)
        self.chunk_tagger.predict(sentence)

        token_list: List[str] = []
        token_tags: List[str] = []

        for token in sentence:
            token_list.append(token.text)

            for label_type in token.annotation_layers.keys():
                # if token.get_labels(label_type)[0].value == "O":
                #     token_tags.append('O')
                # if token.get_labels(label_type)[0].value == "_":
                #     token_tags.append('_')
                token_tags.append(token.get_labels(label_type)[0].value)  # Append token tags for each token

        phrases: List[str] = self._get_flair_phrases(token_list, token_tags)

        return phrases

    @staticmethod
    def _get_flair_phrases(token_list: List[str], token_tags: List[str]):
        """
            Generate the phrases from the extracted tokens and their corresponding tags, by merging the relevant tokens
        :param token_list: List[str]
            List of strings of tokens
        :param token_tags: List[str]
            List of tags in order with tokens, extracted by Flair package
        :return: List[str]
            Returns the list of phrases merging the relevant tokens
        """

        assert len(token_tags) == len(token_list)

        phrases = []
        phrase = ''

        # Creating the list of outside phrases and '_' phrases
        for token, tag in zip(token_list, token_tags):
            if token in string.punctuation:
                continue

            if '-' not in tag:  # '-' do not occur for the single token tags.
                phrases.append(token)

            else:
                state, phrase_pos = tag.split('-')
                if state == 'B':
                    phrase = ''
                    phrase += token
                elif state == 'I':
                    phrase += ' ' + token
                elif state == 'E':
                    phrase += ' ' + token
                    phrases.append(phrase)
                elif state == 'S':
                    phrases.append(token)

        return phrases

    def extract_phrases_tr(self, text: str):
        """
            Returns only noun key phrases
            Source: https://spacy.io/universe/project/spacy-pytextrank

        :param text: The text in which the key phrases should be extracted
        :return: List[str]
            Returns list of strings of key phrases in the text
        """

        doc = self.nlp(text)

        phrases = []

        for p in doc._.phrases:
            phrases.append(p.text)

        return phrases

    @staticmethod
    def tokens_to_str(tokens: List[str]):
        """
            Convert the list of tokens to the string with spaces in the order of list
        :param tokens: List[str]
        :return: str
        """

        token_str = ""
        for token in tokens:
            token_str += token + " "

        return token_str[:-1]

    def is_passive_voice(self, sentence: str):
        """
            Checks if the given sentence has passive voice instances
            # Source: https://gist.github.com/armsp/30c2c1e19a0f1660944303cf079f831a
        :param sentence: str
        :return:  True if the sentence has atleast one passive voice instance
            else False
        """

        matcher = Matcher(self.nlp.vocab)
        doc = self.nlp(sentence)

        passive_rule = [{'DEP': 'nsubjpass', 'OP': '*'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBN'}]
        matcher.add('Passive', None, passive_rule)
        matches = matcher(doc)

        if len(matches) == 0:
            return False
        else:
            return True

    def remove_articles(self, text):
        articles = ["a", "an", "the"]

        doc = self.nlp(text)
        updated = ""

        for token in doc:
            if token.text not in articles:
                updated += token.text + " "

        updated_len = len(updated)

        # Return by removing the last space
        return updated[:updated_len - 1]

    def get_common_keyphrases(self, text1, text2):

        text1_kp = set(self.extract_phrases_tr(text1))
        text2_kp = set(self.extract_phrases_tr(text2))

        text1_updated = set()
        text2_updated = set()

        for text in text1_kp:
            filtered_text = self.remove_articles(text)
            lemmas = self.preprocess.lemmatize(filtered_text)
            filtered_text = self.tokens_to_str(lemmas)
            text1_updated.add(filtered_text)

        for text in text2_kp:
            filtered_text = self.remove_articles(text)
            lemmas = self.preprocess.lemmatize(filtered_text)
            filtered_text = self.tokens_to_str(lemmas)
            text2_updated.add(filtered_text)

        return text1_updated.intersection(text2_updated)

    def open_ie(self, text: str):

        doc = self.nlp(text)
        sentences: List[str] = [sent.text for sent in doc.sents]

        extracted_args = []

        for sentence in sentences:
            extracted_args.extend(self.predictor.predict(sentence=sentence)["verbs"])

        relations = []
        for relation in extracted_args:
            desc = relation["description"]
            relations.append(re.findall("\[(.*?)\]", desc))

        return relations

    def split_by_punct(self, text: str):
        """

        :param text:
        :return:
        """

        return re.split("[?.,:;]", text)

    @staticmethod
    def wordnet_syn(word_1: str):

        word_syn = wordnet.synsets(word_1)

        synonyms = set()

        for syn in word_syn:
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())

        return synonyms

    @staticmethod
    def wordnet_antonym(word_1: str):

        word_syn = wordnet.synsets(word_1)

        antonyms = set()

        for syn in word_syn:
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    for ant in lemma.antonyms():
                        antonyms.add(ant.name().lower())

        return antonyms

    def word_similarity(self, word1: str, word2: str):
        first = self.nlp(word1)
        second = self.nlp(word2)

        return first.similarity(second)

    def get_wordnet_sim(self, word_1: str, word_2: str):

        lemmatizer = WordNetLemmatizer()
        check = lemmatizer.lemmatize(word_2)

        synonyms = self.wordnet_syn(word_1)
        antonyms = self.wordnet_antonym(word_1)

        if check in synonyms:
            return 1
        elif check in antonyms:
            return 0