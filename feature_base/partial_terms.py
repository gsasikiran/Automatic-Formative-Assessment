from typing import List

import spacy

from formative_assessment.utilities.utils import Utilities


class PartialTerms():

    def __init__(self):
        self.utils = Utilities()
        self.nlp = spacy.load('en_core_web_lg')

    def get_missed_phrases(self, question, des_ans, stu_ans):
        """
            Return the missed phrases of student answer expected from the desired answer
        :param question: str
        :param des_ans: str
        :param stu_ans: str

        :return: List[str]
            List of phrases
        """

        des_ans = self.utils.corefer_resolution(des_ans)
        stu_ans = self.utils.corefer_resolution(stu_ans)
        # des_sents = self.utils.split_by_punct(des_ans)
        # stu_sents = self.utils.split_by_punct(stu_ans)

        des_phrases = set()
        stu_phrases = set()

        # for sent in des_sents:

        des_demoted: str = self.utils.demote_ques(question, des_ans)

        if des_demoted:
            des_phrases_softmax: dict = self.utils.softmax_ranked_phrases_rake(des_demoted)
            des_phrases.update(self.utils.extract_phrases_rake(des_demoted))

        # for sent in stu_sents:
        stu_demoted: str = self.utils.demote_ques(question, stu_ans)

        if stu_demoted:
            stu_phrases.update(self.utils.extract_phrases_rake(stu_demoted))

        # Word alignment/Phrase alignment
        missed_phrases = set()

        if des_phrases:
            if stu_phrases:
                aligned_words = align_tokens(list(des_phrases), list(stu_phrases))

                written_phrases = set()
                for value in aligned_words.values():
                    written_phrases.add(value[0])

                missed_phrases = des_phrases - written_phrases

            else:
                missed_phrases = des_phrases

        missed_phrases_score ={}
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
