from typing import List
import numpy as np
import gensim
import tensorflow_hub as hub
from bert_embedding import BertEmbedding
from formative_assessment.utilities.preprocessing import PreProcess


class Embedding:
    def __init__(self):
        self._use_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self._elmo_url = "https://tfhub.dev/google/elmo/3"

        self._use_embed = hub.load(self._use_url)
        self._elmo_embed = hub.load(self._elmo_url)
        self._bert_embed = BertEmbedding()
        self._ft_embed = gensim.models.FastText.load('/home/sgaddi2s/master_thesis/weights/combined/fasttext'
                                                     '/ft').wv

        self.preprocess = PreProcess()

    def bert(self, tokens: List[str]):
        """

        :param tokens:
        :return:
        """
        embedding = self._bert_embed.embedding(sentences=tokens)

        embed_array = []
        for i in range(len(embedding)):
            embed_array.append(embedding[i][1][0])

        return embed_array

    def elmo(self, phrases: List[str]):
        """

        :param tokens:
        :return:
        """
        embeddings = self._elmo_embed(phrases)

        embed_array = []
        for i in range(len(embeddings)):
            embed_array.append(embeddings[i].numpy())

        return embed_array

    def fasttext(self, phrases: List[str], embed_dim = 300):

        phrases_embed = np.zeros((len(phrases), embed_dim))
        for i, phrase in enumerate(phrases):
            tokens = self.preprocess.tokenize(phrase)
            embeddings = np.zeros((len(tokens), embed_dim))

            for j, token in enumerate(tokens):
                embeddings[j] = self._ft_embed[token]
            phrases_embed[i] = self._mowe(embeddings)

        return phrases_embed



    def use(self, tokens: List[str]):
        """

        :param tokens:
        :return:
        """

        embeddings = self._use_embed(tokens)

        embed_array = []
        for i in range(len(embeddings)):
            embed_array.append(embeddings[i].numpy())

        return embed_array

    @staticmethod
    def _sowe(embeddings):
        """
            Compute the sum of word embeddings
        :param embeddings: nd array
            Embeddings of the individual tokens as an array
        :return:
            Sum of the embeddings
        """

        return np.sum(embeddings, axis=0)

    def _mowe(self, embeddings):
        """
            Compute the mean of word embeddings
        :param embeddings: nd array
            Embeddings of the individual tokens as an array
        :return:
            Mean of the embeddings
        """
        return self._sowe(embeddings) / embeddings.shape[0]
