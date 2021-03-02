"""
Abstraction of all the features that are to be extracted for the formative assessment.
"""
import time

import regex as re

from feature_base.terms_interchange import InterchangeOfTerms
from feature_base.wrong_term_identification import IrrelevantTermIdentification
from feature_base.partial_terms import PartialTerms
from formative_assessment.dataset_extractor import DataExtractor
from formative_assessment.utilities.utils import Utilities
from formative_assessment.negated_term_vector import FlipNegatedTermVector

__author__ = "Sasi Kiran Gaddipati"
__credits__ = []
__license__ = ""
__version__ = ""
__last_modified__ = "18.01.2020"
__status__ = "Development"


class FeatureExtractor:
    def __init__(self, question_id: float, student_answer: str, dataset: dict, dir_path: str):

        self.dataset_path = dir_path

        self.dataset_dict = dataset

        self.question_id = question_id
        self.question = dataset[question_id]["question"]
        self.des_ans = dataset[question_id]["desired_answer"]
        self.stu_ans = student_answer

        self.utils = Utilities.instance()
        self.words_score = {}

    def get_irrelevant_terms(self, sem_weight: float = 1, term_threshold: float = 0.35):
        """
            Returns all the probable wrong terms of the student answer

        :param stu_ans: str
        :param sem_weight: float
            Between 0 and 1. Semantic weight we assign to the similarity feature. 1-sim_weight is assigned to the
            lexical feature. default: 0.8
        :param term_threshold: float
            The threshold of which below that value, we consider the term as the wrong term
            default: 0.3

        :return: Set
            Returns the set of wrong terms
        """
        print("Extracting irrelevant terms")
        iti = IrrelevantTermIdentification(self.dataset_dict, DIR_PATH=self.dataset_path)

        pp_des_ans, pp_stu_ans = iti.preprocess(self.question_id, self.stu_ans)
        sim_score = iti.get_sim_score(pp_des_ans, pp_stu_ans)

        lex_weight = 1 - sem_weight
        lex_score = iti.get_lex_score(self.question_id, pp_stu_ans)

        for token in pp_stu_ans:
            self.words_score[token] = (sem_weight * sim_score[token]) + (lex_weight * lex_score[token])

        print("Probable irrelevant terms in the answer")
        irrelevant_terms = {key for (key, value) in self.words_score.items() if value < term_threshold}

        fntv = FlipNegatedTermVector()
        # We demote questions from extracted wrong terms
        terms_demoted = set()
        for term in irrelevant_terms:
            demoted_string = self.utils.demote_ques(self.question, term)
            if demoted_string and not fntv.is_negation(demoted_string):
                terms_demoted.add(demoted_string)

        return terms_demoted

    def is_wrong_answer(self, wrong_answer_threshold: float = -0.5, expected_similarity: float = 0.8):
        """
            Returns if the answer is wrong or not.

        :param wrong_answer_threshold: float
            The float value in between 0 and 1 of which below the value, we consider the answer as the wrong answer
            default: 0.3

        :param expected_similarity: float
            The expected similarity of all the answers

        :return: bool
            Returns true if the answer is totally or sub-optimally correct, else return false
        """

        # Average of the answer score
        total = sum(self.words_score.values()) / (len(self.words_score))

        answer_score = (2 * total) - 1  # normalizing between -1 and 1 from 0 and 1
        print("Answer score: ", answer_score)

        return answer_score < wrong_answer_threshold

    def get_partial_answers(self):

        partial_answers = PartialTerms()

        ques, des_ans = self.utils.combined_coref_res(self.question, self.des_ans)
        _, stu_ans = self.utils.combined_coref_res(self.question, self.stu_ans)

        missed_phrases_all: dict = partial_answers.get_missed_phrases(ques, des_ans, stu_ans)

        # We only extract noun existing phrases, as the phrases like "called explicitly", "whereas needs" will not
        # provide explicit understanding

        missed_phrases_nouns: list = partial_answers.get_noun_phrases(list(missed_phrases_all.keys()))

        missed_phrases = {}
        for phrase in missed_phrases_nouns:
            missed_phrases[phrase] = missed_phrases_all[phrase]

        return missed_phrases

    def get_interchanged_terms(self):
        """
            Prints which terms has been interchanged in the student answer

        :return: List, set
            List: Interchanged terms in the list for each interchanged terms.
            The order of triplets are ["desired topic", "written topic", "written sentence"]
            Set: Missed topics
            The missed topics asked in question and written in desired answer but not presented in the student answer

        """

        iot = InterchangeOfTerms()
        ques, des_ans = self.utils.combined_coref_res(self.question, self.des_ans)
        _, stu_ans = self.utils.combined_coref_res(self.question, self.stu_ans)

        topics = iot.get_topics(ques, des_ans)
        # TODO: if the heads are null, then we assign the best key-phrase as the head and corresponding verbs as the
        #  tree

        iot_dict = {"interchanged": [], "missed_topics": set(), "total_sents_num": 0,  "total_topics": 0}

        if topics:

            des_ans_rel = iot.generate_tree(topics, des_ans)

            # stu_ans = self.utils.corefer_resolution(self.stu_ans)
            stu_ans_rel = iot.generate_tree(topics, stu_ans)

            interchanged, missed_topics = iot.is_interchanged(des_ans_rel, stu_ans_rel)

            total_sents_num = 0
            for topic in stu_ans_rel:
                total_sents_num += total_sents_num + len(stu_ans_rel[topic])

            iot_dict["interchanged"] = interchanged
            iot_dict["missed_topics"] = missed_topics
            iot_dict["total_sents_num"] = total_sents_num
            iot_dict["total_topics"] = len(des_ans_rel)

        return iot_dict
