import re

from formative_assessment.feature_extractor import FeatureExtractor
from formative_assessment.dataset_extractor import ConvertDataType


class AEGrading:
    def __init__(self, qid, stu_answer, dataset, max_score=5):

        self.qid = qid
        self.stu_answer = stu_answer
        self.dataset = dataset
        self.length_ratio = len(dataset[qid]["des_answer"]) / len(stu_answer)
        self.score = max_score
        self.fe = FeatureExtractor(qid, stu_answer, dataset)
        self.wrong_terms = {}
        self.feedback = {}

    def is_not_answered(self):

        if re.match(" *not answered *", self.stu_answer.lower()):
            self.feedback["is_answered"] = "not answered"
            self.score = 0
            return self.score, self.feedback
        else:
            self.feedback["is_answered"] = "answered"
            return self.score, self.feedback

    def wrong_answer_score(self):
        """

        :return:
        """
        is_wrong_answer = self.fe.is_wrong_answer()
        if is_wrong_answer:
            self.feedback["is_wrong_answer"] = "wrong answer"
            self.score = 0
            return self.score, self.feedback
        else:
            self.feedback["is_wrong_answer"] = "not wrong answer"
            return self.score, self.feedback

    def iot_score(self):
        """

        :return:
        """
        interchanged, missed, total_num = self.fe.get_interchanged_terms()

        self.feedback["interchanged"] = interchanged
        self.feedback["missed"] = missed
        # Put feedback here
        if interchanged:
            iot_deduce = len(interchanged) / total_num
            self.score = self.score - iot_deduce

        for each in interchanged:
            self.stu_answer = self.stu_answer - each[2]

        return self.score

    def partial_terms_score(self):
        pass

    def wrong_terms_score(self):
        self.feedback["wrong terms"] = self.fe.get_incorrect_terms()


if __name__ == '__main__':
    PATH = "dataset/mohler/"

    # Convert the data into  dictionary with ids, their corresponding questions, desired answers and student answers
    convert_data = ConvertDataType(PATH)
    dataset_dict = convert_data.to_dict()

    s_no = 8.5
    student_answer = "list because it its size is not determined"
    aeg = AEGrading(s_no, student_answer, dataset_dict)
    aeg.wrong_terms_score()
    aeg.wrong_answer_score()
    aeg.iot_score()
    print(aeg.score)
    print(aeg.feedback)
