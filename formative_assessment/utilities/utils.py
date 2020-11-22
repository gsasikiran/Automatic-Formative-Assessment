from typing import List

import numpy as np
import neuralcoref
import spacy
import pytextrank
import string
import pickle
import tensorflow_hub as hub
from bert_embedding import BertEmbedding
from flair.data import Sentence
from flair.models import SequenceTagger
from scipy.spatial.distance import cosine


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

    def cosine_similarity_matrix(self, des_tokens, stu_tokens):

        des_tokens_array = self.get_use_embed(des_tokens)
        stu_tokens_array = self.get_use_embed(stu_tokens)

        matrix = np.zeros((len(stu_tokens_array), len(des_tokens_array)))

        for i in range(0, len(stu_tokens_array)):
            for j in range(0, len(des_tokens_array)):
                matrix[i][j] = 1 - self.get_cosine_similarity(stu_tokens_array[i], des_tokens_array[j])
        return matrix

    @staticmethod
    def get_frequency(desired_words, total_tokens):
        word_freq = {}

        for word in desired_words:
            count = 0
            for answer in total_tokens:
                if word in answer:
                    count += 1

            word_freq[word] = count

        return word_freq

    def corefer_resolution(self, text):
        neuralcoref.add_to_pipe(self.nlp)
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
        tr = pytextrank.TextRank()
        self.nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)

        doc = self.nlp(text)

        phrases = []

        for p in doc._.phrases:
            phrases.append(p.text)

        return phrases
