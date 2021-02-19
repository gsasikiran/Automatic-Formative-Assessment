from typing import List
import numpy as np
import gensim
import tensorflow as tf
import tensorflow_hub as hub
from bert_embedding import BertEmbedding
from allennlp.modules.elmo import Elmo, batch_to_ids
from formative_assessment.utilities.preprocessing import PreProcess
from formative_assessment.negated_term_vector import FlipNegatedTermVector

class Embedding:

    def __init__(self, name: str):
        self.name = name
        if name == "use":
            self.url = "https://tfhub.dev/google/universal-sentence-encoder/4"
            self._embed = hub.load(self.url)

        elif name == "elmo":

            self.options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            self.weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self._embed = Elmo(self.options_file, self.weight_file, 3, dropout=0)

        elif name == "fasttext":
            self._ft_embed = gensim.models.FastText.load('/home/sgaddi2s/master_thesis/weights/combined/fasttext'
                                                         '/ft').wv
        elif name == "bert":
            self._bert_embed = BertEmbedding()

        else:
            LookupError("We are not currently offering such embeddings called \'", name, "\'")

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

    def elmo(self, phrases: List[str], embed_dim = 1024):
        """

        :param phrases:
        :param embed_dim:
        :return:
        """

        phrases_embed = np.zeros((len(phrases), embed_dim))

        for i, phrase in enumerate(phrases):
            tokens = self.preprocess.tokenize(phrase.lower())
            # We put phrases into [],as elmo considers list inside the list as sentences
            character_ids = batch_to_ids(([tokens]))
            # embeddings are tensors of size [3, 1, len(phrases), 1024].
            # Here 3 refers to number of LSTM layers. 1 refers to number of sentences. 1024 is embedding dimension
            embeddings = self._embed(character_ids)["elmo_representations"][-1].detach().numpy()[0]

            phrases_embed[i] = self._mowe(embeddings)

        return phrases_embed

    def fasttext(self, phrases: List[str], embed_dim=300):
        """

        :param phrases:
        :param embed_dim:
        :return:
        """

        fntv = FlipNegatedTermVector()
        phrases_embed = np.zeros((len(phrases), embed_dim))

        for i, phrase in enumerate(phrases):
            negated_terms = fntv.get_negated_words(phrase)
            tokens = self.preprocess.tokenize(phrase.lower())
            embeddings = np.zeros((len(tokens), embed_dim))

            negation_indices = []
            for j, token in enumerate(tokens):
                if fntv.is_negation(token):
                    negation_indices.append(j)
                    continue
                elif token in negated_terms:
                    embeddings[j] = (-1) * self._ft_embed[token]
                    negated_terms.remove(token)  # We remove to assign negative vector only once.
                else:
                    embeddings[j] = self._ft_embed[token]

            embeddings = np.delete(embeddings, negation_indices, 0)
            phrases_embed[i] = self._mowe(embeddings)
        return phrases_embed

    def use(self, tokens: List[str]):
        """

        :param tokens:
        :return:
        """

        embeddings = self._embed(tokens)

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


class AssignEmbedding(Embedding):

    def __init__(self, embed_name: str):
        super().__init__(embed_name)

    def assign(self, tokens: List[str]):
        if self.name == "use":
            return self.use(tokens)

        elif self.name == "elmo":
            return self.elmo(tokens)

        elif self.name == "fasttext":
            return self.fasttext(tokens)

        else:
            ModuleNotFoundError("Embeddings named \'", self.name, "\' not found")
