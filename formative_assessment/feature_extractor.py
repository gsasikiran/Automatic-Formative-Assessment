"""
Abstraction of all the features that are to be extracted for the formative assessment.
"""

from feature_base.interchange_of_topics import InterchangeOfTopics
from feature_base.irrelevant_terms import IrrelevantTermIdentification
from feature_base.missed_terms import MissedTerms
from formative_assessment.negated_term_vector import FlipNegatedTermVector
from formative_assessment.utilities.utils import Utilities

__author__ = "Sasi Kiran Gaddipati"
__credits__ = []
__license__ = ""
__version__ = ""
__last_modified__ = "10.03.2021"
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

    def get_interchanged_topics(self):
        """
            Extracts interchanged topics and missed topics in the student answer

        :return: Dict
            Returns the dictionary of interchanged topics, missed topics, total relations extracted, and total topics
            extracted

        """

        iot = InterchangeOfTopics()

        # Substituting the pronouns with actual topic
        ques, des_ans = self.utils.combined_coref_res(self.question, self.des_ans)
        _, stu_ans = self.utils.combined_coref_res(self.question, self.stu_ans)

        topics = iot.get_topics(ques, des_ans)

        iot_dict = {"interchanged": [], "missed_topics": set(), "total_relations": 0, "total_topics": 0}

        # When only one topic is asked, the student is expected to write about only that topic. Hence we consider only
        # when multiple topics are present in the question
        if len(topics) > 1:

            # Extracting relations in the desired and student answers from the topic
            des_ans_rel = iot.generate_tree(topics, des_ans)
            stu_ans_rel = iot.generate_tree(topics, stu_ans)

            interchanged, missed_topics = iot.get_interchanged_and_missed(des_ans_rel, stu_ans_rel)

            total_relations = 0
            for topic in stu_ans_rel:
                total_relations += total_relations + len(stu_ans_rel[topic])

            iot_dict["interchanged"] = interchanged
            iot_dict["missed_topics"] = missed_topics
            iot_dict["total_relations"] = total_relations
            # We do not consider the direct topics, as desired answer may not expect all the topics to be answered
            iot_dict["total_topics"] = len(des_ans_rel)

        return iot_dict

    def get_missed_terms(self):
        """
            Generate the missed terms in the student answer comparing it to the desired answer
        :return: List[str]
        """

        print("Extracting missed terms")
        partial_answers = MissedTerms()

        ques, des_ans = self.utils.combined_coref_res(self.question, self.des_ans)
        _, stu_ans = self.utils.combined_coref_res(self.question, self.stu_ans)

        missed_phrases_score: dict = partial_answers.get_missed_phrases(ques, des_ans, stu_ans)

        # We only extract noun existing phrases, as the phrases like "called explicitly", "whereas needs" will not
        # provide explicit understanding

        missed_phrases_nouns: list = partial_answers.get_noun_phrases(list(missed_phrases_score.keys()))

        missed_phrases = {}

        for phrase in missed_phrases_nouns:
            missed_phrases[phrase] = missed_phrases_score[phrase]

        return missed_phrases

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

        words_score = {}
        for token in pp_stu_ans:
            words_score[token] = (sem_weight * sim_score[token]) + (lex_weight * lex_score[token])

        print("Probable irrelevant terms in the answer")
        irrelevant_terms = {key for (key, value) in words_score.items() if value < term_threshold}

        fntv = FlipNegatedTermVector()
        # We demote questions from extracted wrong terms
        terms_demoted = set()
        for term in irrelevant_terms:
            demoted_string = self.utils.demote_ques(self.question, term)
            if demoted_string and not fntv.is_negation(demoted_string):
                terms_demoted.add(demoted_string)

        return terms_demoted


    # def is_wrong_answer(self, wrong_answer_threshold: float = -0.5, expected_similarity: float = 0.8):
    #     """
    #         Returns if the answer is wrong or not.
    #
    #     :param wrong_answer_threshold: float
    #         The float value in between 0 and 1 of which below the value, we consider the answer as the wrong answer
    #         default: 0.3
    #
    #     :param expected_similarity: float
    #         The expected similarity of all the answers
    #
    #     :return: bool
    #         Returns true if the answer is totally or sub-optimally correct, else return false
    #     """
    #
    #     # Average of the answer score
    #     total = sum(self.words_score.values()) / (len(self.words_score))
    #
    #     answer_score = (2 * total) - 1  # normalizing between -1 and 1 from 0 and 1
    #     print("Answer score: ", answer_score)
    #
    #     return answer_score < wrong_answer_threshold
