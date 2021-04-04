"""
 Implementation to extract the missed terms in a student answer automatically.
"""
from typing import List

from formative_assessment.utilities.utils import Utilities, align_tokens

__author__ = "Sasi Kiran Gaddipati"
__credits__ = ["Tim Metzler"]
__license__ = ""
__version__ = "1.0.1"
__email__ = "sasi-kiran.gaddipati@smail.inf.h-brs.de"
__last_modified__ = "04.04.2021"
__status__ = "Prototype"


class MissedTerms:
    def __init__(self):
        self.utils = Utilities.instance()
        self.nlp = self.utils.nlp

    def get_missed_phrases(self, question, des_ans, stu_ans):
        """
            Return the missed phrases of student answer expected from the desired answer
        :param question: str
        :param des_ans: str
        :param stu_ans: str

        :return: Dict
            Dict with phrases as tokens and softmax weights as values
        """

        # des_sents = self.utils.split_by_punct(des_ans)
        # stu_sents = self.utils.split_by_punct(stu_ans)

        des_phrases = set()
        stu_phrases = set()

        # for sent in des_sents:
        des_demoted: str = self.utils.demote_ques(question, des_ans)
        stu_demoted: str = self.utils.demote_ques(question, stu_ans)

        des_phrases_softmax: dict = {}
        if des_demoted:
            des_phrases_softmax = self.utils.softmax_ranked_phrases_rake(des_demoted)
            print("Desired phrases weights:", des_phrases_softmax)
            des_phrases.update(des_phrases_softmax.keys())

        # for sent in stu_sents:
        if stu_demoted:
            stu_phrases.update(self.utils.extract_phrases_rake(stu_demoted))

        # Word alignment/Phrase alignment
        missed_phrases = set()

        if des_phrases:
            if stu_phrases:
                aligned_words = align_tokens(list(des_phrases), list(stu_phrases), align_threshold=0.4)

                written_phrases = set()
                for value in aligned_words.values():
                    written_phrases.add(value[0])

                missed_phrases = des_phrases - written_phrases

            else:
                missed_phrases = des_phrases

        missed_phrases_score = {}
        for phrase in missed_phrases:
            missed_phrases_score[phrase] = des_phrases_softmax[phrase]

        return missed_phrases_score

    def get_noun_phrases(self, phrases: List[str]):
        """
            Return the phrases only if they consists atleast one noun
        :param phrases: List[str]

        :return: List[str]
        """

        noun_phrases = []
        for phrase in phrases:
            doc = self.nlp(phrase)
            for token in doc:
                if token.pos_ == 'NOUN':
                    noun_phrases.append(phrase)
                    break
                else:
                    continue

        return noun_phrases
