import numpy as np
from typing import List

from formative_assessment.utilities.utils import Utilities

class KeyPhraseExtractor:

    def __init__(self):
        self.util = Utilities()

    def _get_cos_sim_norm(self, vector_i, vector_j):
        pass

    def _get_mmr(self, text_vector, phrase_vectors: List, rel_factor=0.5):
        mmr = []
        for i in range(0, len(phrase_vectors)):
            mmr.append(rel_factor * self._get_cos_sim_norm(phrase_vectors[i], text_vector) -  (1 - rel_factor))
        return mmr

    def extract_key_phrases(self, text: str, N = 5, relevance_factor = 0.5):
        """
            Implements "Simple Unsupervised Keyphrase Extraction using Sentence Embeddings" using USE vectors
        :param text:
        :return:
        """

        phrases = self.util.extract_phrases(text)
        informativeness = []

        phrases_use_embed = self.util.get_use_embed(phrases)
        doc_use_embed = self.util.get_use_embed([text])

        for embed in phrases_use_embed:
            informativeness.append(self.util.get_cosine_similarity(embed, doc_use_embed))