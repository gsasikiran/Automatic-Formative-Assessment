from feature_base.wrong_term_identification import WrongTermIdentification
from feature_base.terms_interchange import InterchangeOfTerms
from formative_assessment.dataset_extractor import DataExtractor
from formative_assessment.utilities.utils import Utilities



class FeatureExtractor:
    def __init__(self, question_id: float, stu_answer: str, dataset_path: str= "dataset/mohler/"):

        self.question_id = question_id
        self.dataset_path = dataset_path
        self.extract_data = DataExtractor(dataset_path)

        self.question = self.extract_data.get_questions(question_id)
        self.des_ans = self.extract_data.get_desired_answers(question_id)

        self.stu_ans = stu_answer

        self.words_score = {}

    def get_wrong_terms(self, sim_weight: float = 0.5, wrong_term_threshold: float = 0.2):
        """
            Returns all the probable wrong terms of the student answer
        :param sim_weight: float
            Between 0 and 1. The weight we assign to the similarity feature. 1-sim_weight is assigned to the lexical feature.
        :param wrong_term_threshold: float
            The threshold of which below that value, we consider the term as the wrong term
        :return: Dict
            Returns the dictionary with keys as wrong terms and the corresponding values are their scores
        """

        print("Extracting wrong terms...")
        wti = WrongTermIdentification(self.dataset_path)

        # Preprocessing
        pp_des_ans, pp_stu_ans = wti.preprocess(self.question_id, self.stu_ans)
        print("preprocessing complete")

        # Word alignment/Phrase alignment
        aligned_words = wti.align_tokens(pp_des_ans, pp_stu_ans)
        print("Word alignment: ", aligned_words)

        print("Calculating similarity score")

        # Get Similarity score
        sim_score = wti.get_sim_score(pp_des_ans, pp_stu_ans)

        print("Calculating lexical score")

        # Get Lexical score
        lex_score = wti.get_lex_score(self.question_id, pp_stu_ans)

        # Get lexical weightage
        lex_weight = 1 - sim_weight

        for token in pp_stu_ans:
            self.words_score[token] = (sim_weight * sim_score[token]) + (lex_weight * lex_score[token])

        print("Probable wrong terms in the answer")
        print({k: v for (k, v) in self.words_score.items() if v < wrong_term_threshold})

    def is_wrong_answer(self, wrong_answer_threshold: float = 0.3):
        """
            Prints if the answer is wrong or not.

        :param wrong_answer_threshold: float
            The float value in between 0 and 1 of which below the value, we consider the answer as the wrong answer
        :return: None
        """

        chunks_score = self.words_score

        # Adding up the values of all the chunks
        total = 0
        for phrase in chunks_score:
            total += chunks_score[phrase]

        answer_score = total / len(chunks_score)  # Average of the answer score
        print(answer_score)
        # If the calculated total score is less than given threshold, then we consider that as the wrong_answer
        if answer_score < wrong_answer_threshold:
            print("Wrong answer")
        else:
            print("Not a wrong answer")

    def get_interchanged_terms(self):
        """
            Prints which terms has been interchanged in the student answer
        :return:
        """
        iot = InterchangeOfTerms()
        utils = Utilities()

        heads = iot.get_topics(self.question, self.des_ans)
        # TODO: if the heads are null, then we assign the best keyphrase as the head and corresponding verbs as the tree
        des_ans_rel = iot.generate_tree(heads, self.des_ans)
        stu_ans_rel = iot.generate_tree(heads, self.stu_ans)

        print(des_ans_rel)
        print(stu_ans_rel)







