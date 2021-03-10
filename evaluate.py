"""
    Extracts features of student answer and grades accordingly. Saves the data of all answers in the file as csv.
"""

import datetime
import re
import time

import pandas as pd

from formative_assessment.dataset_extractor import ConvertDataType
from formative_assessment.feature_extractor import FeatureExtractor


class AEGrading:
    """
        Automatically evaluates, grades and provides feedback to students' answers of the datasets.
        Provides feedback as dict including total data of the student answer.
    """

    def __init__(self, qid, stu_answer, dataset, dataset_path, score=5):

        self.qid = qid
        self.stu_answer = stu_answer
        self.dataset = dataset
        self.length_ratio = len(stu_answer) / len(dataset[qid]["desired_answer"])
        self.score = score
        self.fe = FeatureExtractor(qid, stu_answer, dataset, dataset_path)
        self.wrong_terms = {}

        self.feedback = {"id": self.qid, "question": self.dataset[self.qid]["question"],
                         "desired_answer": self.dataset[self.qid]["desired_answer"], "student_answer": stu_answer,
                         "length_ratio": self.length_ratio, "is_answered": "-", "is_wrong_answer": "not wrong answer",
                         "interchanged": "-", "missed_topics": "-", "missed_terms": "-", "irrelevant_terms": "-",
                         "score_avg": 0, "our_score": 0}

    def is_answered(self, default="not answered"):
        """
            Checks if the student answered or not given the default evaluator's string. Assigns score to 'zero' if not
            answered.

        :param default: str
            String to be checked if student not answered
        :return: bool
            True if student answered, else False
        """

        re_string = " *" + default + " *"

        if re.match(re_string, self.stu_answer.lower()):
            self.feedback["is_answered"] = "not answered"
            self.score = 0
            return False

        else:
            self.feedback["is_answered"] = "answered"
            return True

    def iot_score(self):
        """
            Checks if there are any interchange of topics or missed topics and deduce the score accordingly. Deduce
            nothing from the score if there are no interchange of topics or missed topics

        :return: None
        """
        iot = self.fe.get_interchanged_topics()

        interchanged = iot["interchanged"]
        missed_topics = iot["missed_topics"]
        total_relations = iot["total_relations"]
        topics_num = iot["total_topics"]

        self.feedback["interchanged"] = interchanged
        self.feedback["missed_topics"] = missed_topics

        if interchanged:
            iot_deduce = len(interchanged) / total_relations
            self.score = self.score - (iot_deduce * self.score)

        if missed_topics:
            missed_deduce = len(missed_topics) / topics_num
            self.score = self.score - (missed_deduce * self.score)

    def missed_terms_score(self):
        """
            Checks if there are any missed terms in the student answer and deduce score accordingly

        :return: None
        """

        missed_terms = self.fe.get_missed_terms()
        self.feedback["missed_terms"] = missed_terms.keys()

        total = round(sum(missed_terms.values()), 3)
        self.score = self.score - (total * self.score)  # self.score/2

    def irrelevant_terms_score(self):
        """
            Checks if there are any irrelevant terms in the student answer. We do not deduce score for this feature, as
            we consider any irrelevant term as noise.

        :return: None
        """
        self.feedback["irrelevant_terms"] = self.fe.get_irrelevant_terms()


if __name__ == '__main__':

    PATH = "dataset/nn_exam/cleaned/"
    max_score = 2

    # Convert the data into  dictionary with ids, their corresponding questions, desired answers and student answers
    convert_data = ConvertDataType(PATH)
    dataset_dict = convert_data.to_dict()

    id_list = list(dataset_dict.keys())
    data = []

    # random.seed(20)
    for s_no in id_list[1:]:

        # s_no = random.choice(id_list)
        question = dataset_dict[s_no]["question"]
        desired_answer = dataset_dict[s_no]["desired_answer"]

        student_answers = dataset_dict[s_no]["student_answers"]
        scores = dataset_dict[s_no]["scores"]
        # score_me = dataset_dict[s_no]["score_me"]
        # score_other = dataset_dict[s_no]["score_other"]

        for index, _ in enumerate(student_answers):
            # index = random.randint(0, 12)
            start = time.time()
            student_answer = student_answers[index]

            print(s_no, student_answer)
            aeg = AEGrading(s_no, student_answer, dataset_dict, PATH, max_score)

            if aeg.is_answered():
                aeg.iot_score()
                aeg.missed_terms_score()
                aeg.irrelevant_terms_score()
                if aeg.score == 0:
                    aeg.feedback["is_wrong_answer"] = "wrong_answer"

            # aeg.feedback["score_me"] = score_me[index] # Only for mohler data
            # aeg.feedback["score_other"] = score_other[index]
            aeg.feedback["score_avg"] = scores[index]
            aeg.feedback["our_score"] = round((aeg.score * 4)) / 4  # Score in multiples of 0.25

            data.append(aeg.feedback)
            print(aeg.feedback)
            print("It took ", time.time() - start, " secs")
            print("----------------------------------------------------------")

            if len(data) % 50 == 0:
                df = pd.DataFrame(data)
                SAVE_PATH = "outputs/automatic_evaluation/II_NN/" + str(datetime.datetime.now()) + ".csv"
                df.to_csv(SAVE_PATH, sep=",")

    df = pd.DataFrame(data)
    SAVE_PATH = "outputs/automatic_evaluation/II_NN/" + str(datetime.datetime.now()) + ".csv"
    df.to_csv(SAVE_PATH, sep=",")
