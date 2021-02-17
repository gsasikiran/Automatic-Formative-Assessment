import random
import re
import time
import datetime

import pandas as pd

from formative_assessment.feature_extractor import FeatureExtractor
from formative_assessment.dataset_extractor import ConvertDataType


class AEGrading:
    def __init__(self, qid, stu_answer, dataset, max_score=5):

        self.qid = qid
        self.stu_answer = stu_answer
        self.dataset = dataset
        self.length_ratio = len(stu_answer)/len(dataset[qid]["des_answer"])
        self.score = max_score
        self.fe = FeatureExtractor(qid, stu_answer, dataset)
        self.wrong_terms = {}

        self.feedback = {"id": self.qid, "question": self.dataset[self.qid]["question"],
                         "desired_answer": self.dataset[self.qid]["des_answer"], "student_answer": stu_answer,
                         "length_ratio" : self.length_ratio, "is_answered": "-", "is_wrong_answer": "-",
                         "interchanged": "-", "missed_topics": "-", "missed_terms": "-", "incorrect_terms": "-",
                         "assigned_score": 0, "our_score": 0}

    def is_not_answered(self, default="not answered"):

        re_string = " *" + default + " *"
        if re.match(re_string, self.stu_answer.lower()):
            self.feedback["is_answered"] = "not answered"
            self.score = 0
            return True
        else:
            self.feedback["is_answered"] = "answered"
            return False

    def wrong_answer_score(self):
        """

        :return:
        """
        is_wrong_answer = self.fe.is_wrong_answer()
        if is_wrong_answer:
            self.feedback["is_wrong_answer"] = "wrong answer"
            self.score = 0
            return True
        else:
            self.feedback["is_wrong_answer"] = "not wrong answer"
            return False

    def iot_score(self):
        """

        :return:
        """
        interchanged, missed_topics, total_num, topics_num = self.fe.get_interchanged_terms()

        self.feedback["interchanged"] = interchanged
        self.feedback["missed_topics"] = missed_topics
        # Put feedback here
        if interchanged:
            iot_deduce = len(interchanged) / total_num
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

        total = 0
        for term in missed_terms:
            total = total + missed_terms[term]

        self.score = self.score - (total * self.score / 2)

    def wrong_terms_score(self):
        self.feedback["incorrect_terms"] = self.fe.get_irrelevant_terms()
        return self.score, self.feedback


if __name__ == '__main__':
    PATH = "dataset/nn_exam/cleaned/"

    # Convert the data into  dictionary with ids, their corresponding questions, desired answers and student answers
    convert_data = ConvertDataType(PATH)
    dataset_dict = convert_data.to_dict()

    id_list = list(dataset_dict.keys())

    r = random.Random(5)
    data = []

    for i in range(0, 10):

        s_no = r.choice(id_list)
        index = r.randint(0, 12)

        question = dataset_dict[s_no]["question"]
        desired_answer = dataset_dict[s_no]["des_answer"]

        # temporary student answer
        student_answers = dataset_dict[s_no]["stu_answers"]
        scores = dataset_dict[s_no]["scores"]

        # for i, _ in enumerate(student_answers):
        start = time.time()
        student_answer = student_answers[index]

        print(s_no, student_answer)
        aeg = AEGrading(s_no, student_answer, dataset_dict, max_score=2)
        not_answered = aeg.is_not_answered()

        if not not_answered:
            aeg.wrong_terms_score()
            wrong_answer = aeg.wrong_answer_score()  # Wrong answers work only when get_wrong_terms method is run,
            # else returns None
            if not wrong_answer:
                aeg.iot_score()
                aeg.partial_terms_score()

        aeg.feedback["assigned_score"] = scores[index]
        aeg.feedback["our_score"] = aeg.score
        data.append(aeg.feedback)
        print(aeg.feedback)
        print("It took ", time.time() - start, " secs")
        print("----------------------------------------------------------")

    df = pd.DataFrame(data)
    PATH = "outputs/" + str(datetime.datetime.now()) +".csv"
    df.to_csv(PATH, sep=",")

