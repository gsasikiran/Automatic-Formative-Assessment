"""
    Flips the negation implied word vector to the other side. Consider the following example
     text = "I am not using your pen."

     Here, 'not' is negation and 'using' is negation implied word. Hence, we ignore the vector of 'not' and reverse the
     direction of 'using' vector.
"""

__author__ = "Sasi Kiran Gaddipati"
__copyright__ = "Copyright (C) 2021 Sasi Kiran Gaddipati"
__license__ = "Public Domain"
__version__ = "1.0"

import numpy as np
import spacy

from scipy.spatial.distance import cosine


class FlipNegatedTermVector:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self._negations = {"not", "no", "never", "nothing", "neither", "nor", "none", "n't", "cannot", "doesn't", "have'nt"}

    def is_negation(self, token):
        """
            Return whether the given token is negation or not
        :param token: str
        :return: bool
            Returns true, if the token is a negation, else false
        """
        return token in self._negations

    def get_negated_words(self, sent):
        """
            Generate the list of the words that are affected by negation
        :param sent: string
            Input the sentence
        :return: list
            List of words that are directly affected by negation
        """
        # Split the words at negation
        # By observation, it seems that negated terms such as 'no' and 'not' do not affect the preceding words.
        # Instead they affect the succeeding (first occurring noun/pronoun/adjective/verb/adverb) key-word
        words = sent.split()
        negative_word_list = []

        for word in words:
            if self.is_negation(word):
                split_sent_list = sent.split(word + " ")[1:]

                for chunk in split_sent_list:
                    doc = self.nlp(chunk)
                    for token in doc:
                        if token.pos_ == 'NOUN' or token.pos_ == 'PRON' or token.pos_ == 'ADJ' \
                                or token.pos_ == 'VERB' or token.pos_ == 'ADV':
                            negative_word_list.append(token.text)
                            break

        return negative_word_list

    def get_sentence_embed(self, sent, consider_flipping=True):
        """
            Get the sentence embedding through SpaCy
        :param sent: string
            Input the sentence
        :param consider_flipping: bool
            Flip the negated affected word vector or not
        :return: array
            Vector array representing the sentence
        """

        doc = self.nlp(sent)
        negative_words = self.get_negated_words(sent)
        sent_embed = np.zeros(doc[0].vector.size)

        for token in doc:

            # Flipping the word vector into opposite direction
            if consider_flipping and token.text == 'not':
                continue
            if consider_flipping and token.text in negative_words:
                sent_embed -= token.vector

            else:
                sent_embed += token.vector

        return sent_embed


if __name__ == '__main__':
    ref = 'can return a value'

    var_1 = 'cannot return a value'

    fntv = FlipNegatedTermVector()

    ref_embed = fntv.get_sentence_embed(ref)

    var1_wo_flip = fntv.get_sentence_embed(var_1, consider_flipping=False)
    var1_embed = fntv.get_sentence_embed(var_1)
    # var2_embed = fntv.get_sentence_embed(var_2)

    print('Distance between  "%s" and "%s" without flipping: %f' % (ref, var_1, 1 - cosine(ref_embed, var1_wo_flip)))
    print('Distance between "%s" and "%s" with flipping: %f' % (ref, var_1, 1 - cosine(ref_embed, var1_embed)))
