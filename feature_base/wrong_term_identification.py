from math import sqrt
from typing import List, Dict

import numpy as np

from formative_assessment.dataset_extractor import DataExtractor
from formative_assessment.utilities.utils import PreProcess, Utilities


class WrongTermIdentification:
    def __init__(self, PATH='dataset/mohler/'):

        self.PATH = PATH
        self.extract_data = DataExtractor(PATH)
        self.pre_process = PreProcess()
        self.utils = Utilities()

    def preprocess(self, id, student_answer: str, phrases=True):

        question = self.extract_data.get_questions(id)
        des_ans = self.extract_data.get_desired_answers(id)

        des_demoted: str = self.pre_process.demote_ques(question, des_ans)
        stu_demoted: str = self.pre_process.demote_ques(question, student_answer)

        if phrases:

            des_chunks: List[str] = self.utils.extract_phrases(des_demoted)
            stu_chunks: List[str] = self.utils.extract_phrases(stu_demoted)

        else:

            des_chunks = self.pre_process.tokenize(des_demoted)
            stu_chunks = self.pre_process.tokenize(stu_demoted)

        des_filtered = self.pre_process.remove_stopwords(des_chunks)
        stu_filtered = self.pre_process.remove_stopwords(stu_chunks)

        return des_filtered, stu_filtered

    def align_tokens(self, des_tokens: List[str], stu_tokens: List[str]):
        """
            Generate the tuple, to generate the most similar tokens of students answers in the desired answer
        :param des_tokens: List
            List of desired answer's tokens
        :param stu_tokens: List
            List of student answer's tokens
        :return: Dictionary
            Keys: student tokens
            Values: (most similar desired answer token, the cosine similarity between the tokens)
        """

        cos_sim_matrix = self.utils.cosine_similarity_matrix(des_tokens, stu_tokens)

        token_alignment = {}

        for i, column in enumerate(cos_sim_matrix):
            max_sim = max(column)
            index = np.argmax(column)

            token_alignment[stu_tokens[i]] = (des_tokens[int(index)], max_sim)

        return token_alignment

    def _rank_and_sim(self, des_tokens, stu_tokens):

        cos_sim_matrix = self.utils.cosine_similarity_matrix(des_tokens, stu_tokens)
        aligned_tokens: Dict = self.align_tokens(des_tokens, stu_tokens)

        rank_dict = {}
        sim_dict = {}

        for key in aligned_tokens:
            index = stu_tokens.index(key)
            similarity = aligned_tokens[key][1]
            sorted_row = sorted(cos_sim_matrix[index])[::-1]
            rank = (int(np.where(sorted_row == similarity)[0]) + 1) if len(
                np.where(sorted_row == similarity)[0]) == 1 else (int(np.where(sorted_row == similarity)[0][0]) + 1)
            rank_dict[key] = rank
            sim_dict[key] = similarity

        return rank_dict, sim_dict

    def get_sim_score(self, des_tokens, stu_tokens):

        rank_w, sim_w = self._rank_and_sim(des_tokens, stu_tokens)
        score_sw = {}

        for token in stu_tokens:
            score_sw[token] = (1 / rank_w[token]) * sim_w[token]

        return score_sw

    def _get_lex_count(self, student_tokens, id, positive=True):

        stu_answers_id = self.extract_data.get_student_answers(id)
        stu_answers_all = self.extract_data.get_student_answers()
        stu_answers_other = [answer for answer in stu_answers_all if answer not in stu_answers_id]

        tokenized_answers = []
        preprocess = PreProcess()

        if positive:
            for i in range(len(stu_answers_id)):
                tokenized_answers.append(preprocess.tokenize(stu_answers_id[i]))
        else:
            for i in range(len(stu_answers_other)):
                tokenized_answers.append(preprocess.tokenize(stu_answers_other[i]))

        return Utilities().get_frequency(student_tokens, tokenized_answers)

    def get_lex_score(self, id, stu_tokens):

        t_pw = self._get_lex_count(stu_tokens, id, True)
        t_nw = self._get_lex_count(stu_tokens, id, False)

        n_id = len(self.extract_data.get_student_answers(id))
        n_total = len(self.extract_data.get_student_answers())

        score_lw = {}
        # TODO: Remove punctuations or alter considering negative examples
        for token in stu_tokens:
            score = (t_pw[
                         token] / n_id)  # * ((n_total + 1) - n_id) / (t_nw[token] + 1)  # Smoothing factor = 1; as t_nw can be 0
            score_lw[token] = sqrt(score)

        return score_lw
