"""

"""
from typing import List

import spacy


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

    def lemmatize(self, text: str):
        """
            Lemmatize the string of text to the list of SpaCy tokens
        :param text: string
            Input text
        :return: List[str]
            Returns the list of tokens of the input text
        """

        doc = self.nlp(text.lower())
        lemmas = []

        for token in doc:
            lemmas.append(token.lemma_)

        return lemmas

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

        question_tokens = self.lemmatize(question)
        answer_tokens = self.tokenize(answer)  # cannot lemmatize, as we need the original tokens of the answer

        answer_tokens = [token for token in answer_tokens if self.lemmatize(token)[0] not in question_tokens]

        demoted_answer = ''

        for i in range(len(answer_tokens)):
            if i == len(answer_tokens) - 1:
                demoted_answer += answer_tokens[i]
                return demoted_answer
            demoted_answer += answer_tokens[i] + ' '
