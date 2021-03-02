import random
import re
import time
import datetime

import pandas as pd

from formative_assessment.feature_extractor import FeatureExtractor
from formative_assessment.dataset_extractor import ConvertDataType


class AEGrading:
    def __init__(self, qid, stu_answer, dataset, dataset_path, max_score=5):

        self.qid = qid
        self.stu_answer = stu_answer
        self.dataset = dataset
        self.length_ratio = len(stu_answer) / len(dataset[qid]["desired_answer"])
        self.score = max_score
        self.fe = FeatureExtractor(qid, stu_answer, dataset, dataset_path)
        self.wrong_terms = {}

        self.feedback = {"id": self.qid, "question": self.dataset[self.qid]["question"],
                         "desired_answer": self.dataset[self.qid]["des_answer"], "student_answer": stu_answer,
                         "length_ratio": self.length_ratio, "is_answered": "-", "is_wrong_answer": "not wrong answer",
                         "interchanged": "-", "missed_topics": "-", "missed_terms": "-", "irrelevant_terms": "-", "score_avg": 0,
                         "our_score": 0}

    def is_answered(self, default="not answered"):

        re_string = " *" + default + " *"
        if re.match(re_string, self.stu_answer.lower()):
            self.feedback["is_answered"] = "not answered"
            self.score = 0
            return False
        else:
            self.feedback["is_answered"] = "answered"
            return True

    # def wrong_answer_score(self):
    #     """
    #
    #     :return:
    #     """
    #     is_wrong_answer = self.fe.is_wrong_answer()
    #     if is_wrong_answer:
    #         self.feedback["is_wrong_answer"] = "wrong answer"
    #         self.score = 0
    #         return True
    #     else:
    #         self.feedback["is_wrong_answer"] = "not wrong answer"
    #         return False

    def iot_score(self):
        """

        :return:
        """
        iot = self.fe.get_interchanged_terms()

        interchanged = iot["interchanged"]
        missed_topics = iot["missed_topics"]
        total_sents = iot["total_sents_num"]
        topics_num = iot["total_topics"]

        self.feedback["interchanged"] = interchanged
        self.feedback["missed_topics"] = missed_topics
        # Put feedback here
        if interchanged:
            iot_deduce = len(interchanged) / total_sents
            self.score = self.score - (iot_deduce * self.score)
        #
        # for each in interchanged:
        #     self.stu_answer = self.stu_answer - each[2]
        if missed_topics:
            missed_deduce = len(missed_topics) / topics_num
            self.score = self.score - (missed_deduce * self.score)

        return self.score

    def partial_terms_score(self):

        # terms_score = self._softmax_ranked_phrases_rake(dataset_dict[self.qid]["des_answer"])

        missed_terms = self.fe.get_partial_answers()

        self.feedback["missed_terms"] = missed_terms.keys()

        total = round(sum(missed_terms.values()), 3)
        self.score = self.score - (total * self.score)  # self.score/2

    def irrelevant_terms_score(self):
        self.feedback["irrelevant_terms"] = self.fe.get_irrelevant_terms()
        # return self.score, self.feedback


if __name__ == '__main__':
    PATH = "dataset/mohler/cleaned/"

    # Convert the data into  dictionary with ids, their corresponding questions, desired answers and student answers
    convert_data = ConvertDataType(PATH)
    dataset_dict = convert_data.to_dict()

    id_list = list(dataset_dict.keys())


    data = []

    for s_no in id_list[5:]:

        # s_no = r.choice(id_list)
        # index = r.randint(0, 12)

        question = dataset_dict[s_no]["question"]
        desired_answer = dataset_dict[s_no]["desired_answer"]

        # temporary student answer
        student_answers = dataset_dict[s_no]["student_answers"]
        scores = dataset_dict[s_no]["scores"]
        score_me = dataset_dict[s_no]["score_me"]
        score_other = dataset_dict[s_no]["score_other"]

        for index, _ in enumerate(student_answers):

            start = time.time()
            student_answer = student_answers[index]

            print(s_no, student_answer)
            aeg = AEGrading(s_no, student_answer, dataset_dict, PATH, max_score=5,)

            if aeg.is_answered():
                aeg.iot_score()
                aeg.partial_terms_score()
                aeg.irrelevant_terms_score()
                if aeg.score == 0:
                    aeg.feedback["is_wrong_answer"] = "wrong_answer"

            aeg.feedback["score_me"] = score_me[index]
            aeg.feedback["score_other"] = score_other[index]
            aeg.feedback["score_avg"] = scores[index]
            aeg.feedback["our_score"] = aeg.score

            data.append(aeg.feedback)
            print(aeg.feedback)
            print("It took ", time.time() - start, " secs")
            print("----------------------------------------------------------")

            if len(data) % 50 == 0:

                df = pd.DataFrame(data)
                PATH = "outputs/automatic_evaluation/I/" + str(datetime.datetime.now()) + ".csv"
                df.to_csv(PATH, sep=",")

    df = pd.DataFrame(data)
    PATH = "outputs/automatic_evaluation/I/" + str(datetime.datetime.now()) + ".csv"
    df.to_csv(PATH, sep=",")