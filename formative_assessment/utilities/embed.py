from typing import List

import tensorflow_hub as hub
from bert_embedding import BertEmbedding


class Embedding:
    def __init__(self):
        self._use_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self._elmo_url = "https://tfhub.dev/google/elmo/3"

        self._use_embed = hub.load(self._use_url)
        self._elmo_embed = hub.load(self._elmo_url)

        self._bert_embed = BertEmbedding()

    def bert(self, tokens: List[str]):
        embedding = self._bert_embed.embedding(sentences=tokens)

        embed_array = []
        for i in range(len(embedding)):
            embed_array.append(embedding[i][1][0])

        return embed_array

    def elmo(self, tokens: List[str]):
        embeddings = self._elmo_embed(tokens)

        embed_array = []
        for i in range(len(embeddings)):
            embed_array.append(embeddings[i].numpy())

        return embed_array

    def use(self, tokens: List[str]):

        embeddings = self._use_embed(tokens)

        embed_array = []
        for i in range(len(embeddings)):
            embed_array.append(embeddings[i].numpy())

        return embed_array
