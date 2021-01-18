"""
Abstraction of all the features that are to be extracted for the formative assessment.
"""
from feature_base.terms_interchange import InterchangeOfTerms
from feature_base.wrong_term_identification import WrongTermIdentification
from formative_assessment.dataset_extractor import DataExtractor
from formative_assessment.utilities.utils import Utilities

__author__ = "Sasi Kiran Gaddipati"
__credits__ = []
__license__ = ""
__version__ = ""
__last_modified__ = "06.01.2020"
__status__ = "Development"


class FeatureExtractor:
    def __init__(self, question_id: float, stu_answer: str, dataset: dict, dir_path: str = "dataset/mohler/"):

        self.question_id = question_id
        self.dataset_path = dir_path
        self.extract_data = DataExtractor(self.dataset_path)
        self.dataset_dict = dataset
        self.question = self.dataset_dict[question_id]["question"]
        self.des_ans = self.dataset_dict[question_id]["des_answer"]

        self.stu_ans = stu_answer

        self.words_score = {}

        self.utils = Utilities()
        self.wti = WrongTermIdentification(self.dataset_dict, DIR_PATH=self.dataset_path)

    def get_wrong_terms(self, sem_weight: float = 0.5, wrong_term_threshold: float = 0.4):
        """
            Returns all the probable wrong terms of the student answer

        :param sem_weight: float
            Between 0 and 1. Semantic weight we assign to the similarity feature. 1-sim_weight is assigned to the
            lexical feature. default: 0.5
        :param wrong_term_threshold: float
            The threshold of which below that value, we consider the term as the wrong term
            default: 0.4

        :return: Dict
            Returns the dictionary with keys as wrong terms and the corresponding values are their scores
        """

        print("Extracting wrong terms...")

        # Preprocessing
        pp_des_ans, pp_stu_ans = self.wti.preprocess(self.question_id, self.stu_ans)
        print("preprocessing complete")

        # Word alignment/Phrase alignment
        aligned_words = self.wti.align_tokens(pp_des_ans, pp_stu_ans)
        print("Word alignment: ", aligned_words)

        print("Calculating similarity score")
        # Get Similarity score
        sim_score = self.wti.get_sim_score(pp_des_ans, pp_stu_ans)

        print("Calculating lexical score")
        # Get lexical weightage
        lex_weight = 1 - sem_weight

        # Get Lexical score
        lex_score = self.wti.get_lex_score(self.question_id, pp_stu_ans)

        for token in pp_stu_ans:
            self.words_score[token] = (sem_weight * sim_score[token]) + (lex_weight * lex_score[token])

        print("Probable wrong terms in the answer")
        print({k: v for (k, v) in self.words_score.items() if v < wrong_term_threshold})

    def is_wrong_answer(self, wrong_answer_threshold: float = 0.25, expected_similarity: float = 0.8):
        """
            Returns if the answer is wrong or not.

        :param wrong_answer_threshold: float
            The float value in between 0 and 1 of which below the value, we consider the answer as the wrong answer
            default: 0.3

        :return: bool
            Returns true if the answer is totally or sub-optimally correct, else return false
        """

        chunks_score = self.words_score

        # Adding up the values of all the chunks
        total = 0
        for phrase in chunks_score:
            total += chunks_score[phrase]

        answer_score = total / (expected_similarity * len(chunks_score))  # Average of the answer score
        print("Answer score: ", answer_score)
        # If the calculated total score is less than given threshold, then we consider that as the wrong_answer

        if answer_score < wrong_answer_threshold:
            print("Wrong answer")
        else:
            print("Not a wrong answer")

        return answer_score > wrong_answer_threshold

    def get_suboptimal_answers(self):

        des_sents = self.utils.split_by_punct(self.des_ans)
        stu_sents = self.utils.split_by_punct(self.stu_ans)

        des_phrases = set()
        stu_phrases = set()

        for sent in des_sents:

            des_demoted: str = self.utils.demote_ques(self.question, sent)
            if des_demoted:
                des_phrases.update(self.utils.extract_phrases_tr(des_demoted))

        for sent in stu_sents:
            stu_demoted: str = self.utils.demote_ques(self.question, sent)

            if stu_demoted:
                stu_phrases.update(self.utils.extract_phrases_tr(stu_demoted))

        # Word alignment/Phrase alignment
        missed_phrases = set()

        if des_phrases:
            if stu_phrases:
                #TODO: get if aligned is not greater than some similarity (say 0.5), assign not aligned.
                aligned_words = self.wti.align_tokens(list(des_phrases), list(stu_phrases))

                written_phrases = set()

                for value in aligned_words.values():
                    written_phrases.add(value[0])

                missed_phrases = written_phrases - des_phrases

            else:
                missed_phrases = des_phrases

        print("Unanswered topics")

        if missed_phrases:
            print("The student didn't mention about: ")
            print(missed_phrases)
        else:
            print("You have written about all the topics")

    def get_interchanged_terms(self):
        """
            Prints which terms has been interchanged in the student answer

        :return:
        """
        iot = InterchangeOfTerms()

        heads = iot.get_topics(self.question, self.des_ans)
        # TODO: if the heads are null, then we assign the best key-phrase as the head and corresponding verbs as the
        #  tree
        des_ans_rel = iot.generate_tree(heads, self.des_ans)
        stu_ans = self.utils.corefer_resolution(self.stu_ans)
        stu_ans_rel = iot.generate_tree(heads, stu_ans)

        iot.is_interchanged(des_ans_rel, stu_ans_rel)
