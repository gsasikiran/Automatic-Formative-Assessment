""" Creates various helper function for preprocessing and processing.

PreProcess class consists various functions such as tokenization, normalization etc.,
Utilities consists of the helper functions for processing various functions, that do not
go into any other classes.
"""

from typing import List

import numpy as np
import neuralcoref
import spacy
import pytextrank
import string
import pickle
import tensorflow_hub as hub
from bert_embedding import BertEmbedding
from spacy.matcher import Matcher
from flair.data import Sentence
from flair.models import SequenceTagger
from scipy.spatial.distance import cosine

__author__ = "Sasi Kiran Gaddipati"
__credits__ = []
__license__ = ""
__version__ = ""
__last_modified__ = "23.11.2020"
__status__ = "Development"


class PreProcess:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    @staticmethod
    def normalize_case(text: str):
        """
            Normalizes the text to the lower case
        :param text: string
        :return: string
            Returns the string in the lower case
        """
        return text.lower()

    def tokenize(self, text: str):
        """
            Tokenize the string of text to the list of SpaCy tokens
        :param text: string
            Input text
        :return: List[str]
            Returns the list of tokens of the input text
        """
        doc = self.nlp(text.lower())
        tokens = [token.text for token in doc if not token.is_punct]
        return tokens

    def remove_stopwords(self, tokens: List[str]):
        """
            Remove stopwords in the tokens of text
        :param tokens: List[str]
            List of string of tokens
        :return: List[str]
            Returns the list of tokens with removing stopwords

        # Source:
        # https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/
        """
        filtered_tokens = []

        for token in tokens:
            lexeme = self.nlp.vocab[token]
            if not lexeme.is_stop:
                filtered_tokens.append(token)
        return filtered_tokens

    def demote_ques(self, question: str, answer: str):
        """
            Removes the tokens of the answer repeated from the question
        :param question: string
        :param answer: string
        :return: string
            Returns string of answer with removing the words present in the question
        """

        question_tokens = self.tokenize(question)
        answer_tokens = self.tokenize(answer)
        answer_tokens = [token for token in answer_tokens if token not in question_tokens]

        demoted_answer = ''

        for i in range(len(answer_tokens)):
            if i == len(answer_tokens) - 1:
                demoted_answer += answer_tokens[i]
                return demoted_answer
            demoted_answer += answer_tokens[i] + ' '


class Utilities:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        neuralcoref.add_to_pipe(self.nlp)
        tr = pytextrank.TextRank()
        self.nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)

    @staticmethod
    def get_use_embed(tokens):

        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        embed = hub.load(module_url)
        embeddings = embed(tokens)
        word_array = []
        for i in range(len(embeddings)):
            word_array.append(embeddings[i].numpy())
        return word_array

    @staticmethod
    def _get_bert_embed(tokens):

        embedding = BertEmbedding().embedding(sentences=tokens)

        word_array = []
        for i in range(len(embedding)):
            word_array.append(embedding[i][1][0])
        return word_array

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
        return cosine(array_1, array_2)

    def cosine_similarity_matrix(self, tokens_a: List[str], tokens_b: List[str]):
        """
            Creates a matrix with similarity values from USE embeddings for each token in text_a to each word in text_b
        :param tokens_a: List[str]
             Tokens of text
        :param tokens_b:
            Tokens of text
        :return: np array
            Returns the matrix with similarity values
        """

        tokens_a_array = self.get_use_embed(tokens_a)
        tokens_b_array = self.get_use_embed(tokens_b)

        matrix = np.zeros((len(tokens_b_array), len(tokens_a_array)))

        for i in range(0, len(tokens_b_array)):
            for j in range(0, len(tokens_a_array)):
                matrix[i][j] = 1 - self.get_cosine_similarity(tokens_b_array[i], tokens_a_array[j])
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
        tagger = SequenceTagger.load('chunk')
        tagger.predict(sentence)

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

    def is_passive_voice(self, sentence: str):
        # Source: https://gist.github.com/armsp/30c2c1e19a0f1660944303cf079f831a

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
            text1_updated.add(self.remove_articles(text))

        for text in text2_kp:
            text2_updated.add(self.remove_articles(text))

        return text1_updated.intersection(text2_updated)
