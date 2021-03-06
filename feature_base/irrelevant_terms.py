"""
 Implementation to extract the wrong terms/phrases in a student answer automatically.
"""
from typing import List, Dict

import numpy as np

from formative_assessment.dataset_extractor import DataExtractor
from formative_assessment.utilities.utils import Utilities, cosine_sim_matrix, align_tokens

__author__ = "Sasi Kiran Gaddipati"
__credits__ = ["Tim Metzler"]
__license__ = ""
__version__ = "1.0.1"
__email__ = "sasi-kiran.gaddipati@smail.inf.h-brs.de"
__last_modified__ = "04.04.2021"
__status__ = "Prototype"


class IrrelevantTermIdentification:
    def __init__(self, dataset: dict, DIR_PATH: str):

        self.PATH = DIR_PATH
        self.extract_data = DataExtractor(DIR_PATH)
        self.dataset_dict = dataset

        self.utils = Utilities.instance()

        self.cos_sim_matrix = np.array([])

    def preprocess(self, qid, student_answer: str, get_phrases=True):

        """
         Preprocessing pipeline for the wrong term extraction

        :param qid: float/int
            Identity of questions
        :param student_answer: str
        :param get_phrases: bool
            Extracts phrase-wise, if true, else extracts token-wise
            default: True

        :return: List[str], List[str]
            List of desired answer tokens, List of student answer tokens
        """

        # question: str = self.dataset_dict[qid]["question"]
        des_ans: str = self.dataset_dict[qid]["desired_answer"]

        des_chunks: List[str] = []
        stu_chunks: List[str] = []

        # Phrase extraction
        if get_phrases:
            des_chunks = self.utils.extract_phrases_rake(des_ans)
            stu_chunks = self.utils.extract_phrases_rake(student_answer)

        # Tokenization
        else:
            # Remove articles
            des_filtered = self.utils.remove_articles(des_ans)
            stu_filtered = self.utils.remove_articles(student_answer)

            # Splitting answers by
            des_sents = self.utils.split_by_punct(des_filtered)
            stu_sents = self.utils.split_by_punct(stu_filtered)

            for sent in des_sents:
                des_chunks.extend(self.utils.tokenize(sent))

            for sent in stu_sents:
                stu_chunks.extend(self.utils.tokenize(sent))

            des_chunks = self.utils.remove_stopwords(des_chunks)
            stu_chunks = self.utils.remove_stopwords(stu_chunks)

        return des_chunks, stu_chunks

    def _rank_and_sim(self, des_tokens: List[str], stu_tokens: List[str]):
        """
           Returns the rank of student tokens and similarity of student tokens

        :param des_tokens: List[str]
            Desired answer tokens/phrases
        :param stu_tokens: List[str]
            Student answer tokens/phrases

        :return: dict, dict
            Dict of student answers with keys of tokens (str) and values with the ranks (int)
            Dict of student answers with keys of tokens (str) and values with their max similarities (float)
        """

        # NOTE: we consider student tokens as rows and desired tokens as columns for ease of operation
        self.cos_sim_matrix = cosine_sim_matrix(stu_tokens, des_tokens)
        print(self.cos_sim_matrix, stu_tokens, des_tokens)
        aligned_tokens: Dict = align_tokens(des_tokens, stu_tokens, align_threshold=-1)

        rank_dict = {}
        sim_dict = {}

        for key in aligned_tokens:

            # stu_token_idx = stu_tokens.index(key)
            des_token_idx = aligned_tokens[key][2]
            max_sim = aligned_tokens[key][1]  # max_similarity
            # des_token_idx = np.argmax(self.cos_sim_matrix[stu_token_idx])

            sorted_row = sorted(self.cos_sim_matrix.T[des_token_idx])[::-1]
            index = np.where(sorted_row == max_sim)

            if len(index[0]) == 1:
                rank = (int(index[0]) + 1)  # changing zero index to one index
            else:
                rank = (int(index[0][0]) + 1)  # changing zero index to one index

            rank_dict[key] = rank
            sim_dict[key] = max_sim

        return rank_dict, sim_dict

    def get_sim_score(self, des_tokens, stu_tokens):
        """
            Return similarity score of the tokens. Calculate using the formula (1/rank_token) * (similarity_token)

        :param des_tokens: List[str]
            Desired answer tokens/phrases
        :param stu_tokens: List[str]
            Student answer tokens/phrases

        :return: dict
            Keys: str = Student answer tokens/phrases
            Values: float = semantic similarity score of the student tokens/phrases
        """

        rank_w, sim_w = self._rank_and_sim(des_tokens, stu_tokens)
        score_sw = {}

        # TODO: Rank can be 1+0.2 instead of 2. That will decrease the difference of ranks while you can also try logs.

        for token in stu_tokens:
            score_sw[token] = (1 / rank_w[token]) * sim_w[token]

        return score_sw

    def _get_lex_count(self, id, student_tokens: List[str] = [], positive=True):
        """
        Count the frequency of tokens in the total answers

        :param id: float/int/str (usually)
            Unique identity of the question
        :param student_tokens: List[str]
            Student answer tokens/phrases
        :param positive: bool
            if true, returns number of student answers, the tokens have appeared
            if false, returns number of student answers, the tokens have not appeared

        :return: Dict
            Keys: (str) tokens
            Values: (int) count
        """

        stu_answers_id = self.dataset_dict[id]["student_answers"]

        tokenized_answers = []

        if positive:
            for i in range(len(stu_answers_id)):
                tokenized_answers.append(self.utils.extract_phrases_rake(str(stu_answers_id[i])))
        else:
            # Extracting the other student answers, that do not belong to given id. These are used to check the false
            # positives of the tokens
            stu_answers_all = self.extract_data.get_student_answers()
            stu_answers_other = [answer for answer in stu_answers_all if answer not in stu_answers_id]
            for i in range(len(stu_answers_other)):
                tokenized_answers.append(self.utils.extract_phrases(stu_answers_other[i]))

        return self.utils.get_frequency(student_tokens, tokenized_answers)

        # return collections.Counter(tokenized_answers)

    def get_lex_score(self, id, stu_tokens):
        """
        Calculates the lexical scores. The formula is (number of occurrences of chunk/total number of answers for the given id)

        :param id: float/int/str (usually)
            Unique identity of the question
        :param stu_tokens:List[str]
            Student answer tokens/phrases

        :return: dict
            Keys: (str) Tokens/phrases
            Values: (float) lexical score
        """

        t_pw = self._get_lex_count(id, stu_tokens, True)
        # t_nw = self._get_lex_count(id, stu_tokens, False)

        n_id = len(self.dataset_dict[id]["student_answers"])
        # n_total = len(self.extract_data.get_student_answers())

        score_lw = {}
        # TODO: Remove punctuations or alter considering negative examples
        for token in stu_tokens:
            score = (t_pw[token] - 1) / (
                        n_id - 1)  # * ((n_total + 1) - n_id) / (t_nw[token] + 1)  # Smoothing factor = 1; as t_nw can be 0
            # score_lw[token] = sqrt(score)
            score_lw[token] = score

        return score_lw

    # def get_softmax_sim(self, des_tokens, stu_tokens):
    #     """
    #
    #     :param des_tokens:
    #     :param stu_tokens:
    #     :return:
    #     """
    #
    #     self.cos_sim_matrix = cosine_sim_matrix(stu_tokens, des_tokens)
    #     softmax_matrix = softmax(self.cos_sim_matrix, axis=1)  # Column axis
    #
    #     weighted_matrix = np.sqrt(np.multiply(self.cos_sim_matrix, softmax_matrix))
    #     sim_matrix = np.average(weighted_matrix, axis=1)
    #
    #     sim_score_dict = {}
    #
    #     # assert len(stu_tokens) ==  sim_matrix.size
    #
    #     for i in range(0, len(stu_tokens)):
    #         sim_score_dict[stu_tokens[i]] = sim_matrix[i]
    #
    #     return sim_score_dict
