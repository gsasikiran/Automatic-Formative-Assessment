import pandas as pd


class DataExtractor:
    def __init__(self, PATH: str):

        self.PATH = PATH
        self.ques_df = pd.read_csv(PATH + "questions.csv", error_bad_lines=False, delimiter="\t")
        self.stu_ans_df = pd.read_csv(PATH + "answers.csv", error_bad_lines=False, delimiter="\t")

    def get_questions(self, ques_id: float = None):
        """
        The column name should be named as the question in questions ile
        :param ques_id: float
            The question id is float
        :return: List[str] / str
            Return the list of all questions if no id is given,
            else return the string of the question for the given ques_id
        """

        if ques_id is not None:
            question_list = self.ques_df.loc[self.ques_df["id"] == ques_id, "question"].iloc[0]
            return question_list

        return self.ques_df.question.to_list()

    def get_desired_answers(self, solution_id: float = None):
        """
        The column name of the desired answer should be named as the solution in questions file
        :param solution_id: float
        The id of the desired answer to be returned
        :return: List[str] / str
        Return the list of all solutions if no id is given,
        else return the string of the solution for the given solution_id
        """

        if solution_id is not None:
            des_ans_list = self.ques_df.loc[self.ques_df["id"] == solution_id, "solution"].iloc[0]
            return des_ans_list

        return self.ques_df.solution.to_list()

    def get_student_answers(self, stu_ans_id=None):
        """
        The column name of the student answer should be named as the answer in answers file
        :param stu_ans_id: float
            The id of the the student answers to get
        :return: List[str]
            Returns the list of all student answers,
            Else if the stu_ans_id is given, returns the list of student answers for that id
        """

        if stu_ans_id is not None:
            stu_ans_list = list(self.stu_ans_df.loc[self.stu_ans_df["id"] == stu_ans_id, "answer"])
            return stu_ans_list

        return self.stu_ans_df.answer.to_list()
