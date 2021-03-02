"""
 Consists dataset utilities, that extract the data from csv, pickle and json files. Converts dataset types and files.
"""
import csv
import json
import pickle

import pandas as pd

from formative_assessment.utilities.utils import Utilities

__author__ = "Sasi Kiran Gaddipati"
__credits__ = []
__license__ = ""
__version__ = ""
__last_modified__ = "06.12.2020"
__status__ = "Development"


class DataExtractor:
    def __init__(self, PATH: str):

        self.PATH = PATH

        self.ques_df = pd.read_csv(PATH + "questions.csv", error_bad_lines=False, delimiter="\t")
        self.stu_ans_df = pd.read_csv(PATH + "answers.csv", error_bad_lines=False, delimiter="\t")

        self.utils = Utilities.instance()

    def get_questions(self, ques_id: float = None):
        """
        The column name should be named as the question in questions file

        :param ques_id: float
            The question id is float
            default: None

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
            default: None

        :return: List[str] / str
        Return the list of all solutions if no id is given,
        else return the string of the solution for the given solution_id
        """

        if solution_id is not None:
            des_ans_list = self.ques_df.loc[self.ques_df["id"] == solution_id, "solution"].iloc[0]
            return des_ans_list

        return self.ques_df.solution.to_list()

    def get_student_answers(self, stu_ans_id: float = None):
        """
        The column name of the student answer should be named as the answer in answers file

        :param stu_ans_id: float
            The id of the the student answers to get
            default: None

        :return: List[str]
            Returns the list of all student answers,
            Else if the stu_ans_id is given, returns the list of student answers for that id
        """

        if stu_ans_id is not None:
            stu_ans_list = list(self.stu_ans_df.loc[self.stu_ans_df["id"] == stu_ans_id, "answer"])
            return stu_ans_list

        return self.stu_ans_df.answer.to_list()

    def get_scores(self, stu_ans_id: float = None):
        """
        The column name of the student answer should be named as the answer in answers file

        :param stu_ans_id: float
            The id of the the student answers to get
            default: None

        :return: List[int]
            Returns the list of all student scores,
            Else if the stu_ans_id is given, returns the list of student scores for that id
        """

        if stu_ans_id is not None:
            stu_ans_list = list(self.stu_ans_df.loc[self.stu_ans_df["id"] == stu_ans_id, "score_avg"])
            return stu_ans_list

        return self.stu_ans_df.score_avg.to_list()

    def other_mohler_scores(self,stu_ans_id: float = None):
        """
        The column name of the student answer should be named as the answer in answers file

        :param stu_ans_id: float
            The id of the the student answers to get
            default: None

        :return: List[int]
            Returns the list of all student scores,
            Else if the stu_ans_id is given, returns the list of student scores for that id
        """

        if stu_ans_id is not None:
            stu_ans_list_me = list(self.stu_ans_df.loc[self.stu_ans_df["id"] == stu_ans_id, "score_me"])
            stu_ans_list_other = list(self.stu_ans_df.loc[self.stu_ans_df["id"] == stu_ans_id, "score_other"])
            return {"score_me": stu_ans_list_me, "score_other" : stu_ans_list_other}

        return {"score_me": self.stu_ans_df.score_me.to_list(), "score_other" : self.stu_ans_df.score_other.to_list()}

    # def from_pickle(self, file_rel_path: str = ""):
    #     """
    #      Reads the dictionary dataset from pickle file
    #
    #     :param file_rel_path: str
    #         File name of the pickle in the directory PATH
    #         default: empty string ("")
    #
    #     :return:
    #         Returns the data from pickle
    #     """
    #
    #     file_path = self.PATH + file_rel_path
    #
    #     with open(file_path, "rb") as pfile:
    #         dataset = pickle.load(pfile)
    #
    #     return dataset

    # def crawl_phrases(self):
    #     """
    #     Crawl all the phrases from the dataset i.e. from all questions, desired answers and student answers. This assists
    #     in saving embeddings in a file for all the phrases.
    #
    #     :return: set()
    #         Set of all the unique phrases
    #     """
    #
    #     sent_list = []
    #
    #     sent_list.extend(self.get_questions())
    #     sent_list.extend(self.get_desired_answers())
    #     sent_list.extend(self.get_student_answers())
    #
    #     phrases = set()
    #
    #     for sentence in sent_list:
    #         phrases.update(self.utils.extract_phrases(sentence))
    #
    #     return phrases


class ConvertDataType(DataExtractor):

    def __init__(self, PATH: str):
        super().__init__(PATH)

    def to_dict(self):
        """
        Convert the dataframe of pandas into dict

        :return: dict Returns the dictionary in the order of { id_1 : {"question" : "", "des_answer" : "",
        "stu_answers" : ["", ""]}, id_2 : {"question" : "", "des_answer" : "", "stu_answers" : [ "", ""]}, ---}
        """
        data = {}

        # Generate the list of ids from question dataframe
        id_list = self.ques_df["id"].to_list()

        # For each id, generate corresponding, question, desired answer and list of student answers
        for id in id_list:

            data[id] = {}
            data[id]["question"] = self.get_questions(id)
            data[id]["des_answer"] = self.get_desired_answers(id)
            data[id]["stu_answers"] = self.get_student_answers(id)
            data[id]["scores"] = self.get_scores(id)
            data[id]["score_me"] = self.other_mohler_scores(id)["score_me"]
            data[id]["score_other"] = self.other_mohler_scores(id)["score_other"]

        return data

    # def csv_to_pickle(self, dataset_name: str = "pickled"):
    #     """
    #     Save the dataframe of csv into the dict and save into the pickle. The dict is in the form as explained
    #     in the documentation of :func:`<dataset_extractor.ConvertDataType().to_dict>`
    #
    #     :param dataset_name: str
    #         The name to be saved with in the directory PATH
    #         default: json
    #
    #     :return: None
    #
    #     """
    #     dataset = self.to_dict()
    #     save_path = self.PATH + dataset_name + "_data.p"
    #     pickle.dump(dataset, open(save_path, "wb"))

    # def json_to_pickle(self, dataset_name: str = "json"):
    #     """
    #     Save the dataframe of csv into the dict and save into the json file. The dict is in the form as explained
    #     in the documentation of :func:`<dataset_extractor.ConvertDataType().to_dict>`
    #
    #     :param dataset_name: str
    #         The name to be saved with in the directory PATH
    #         default: json
    #
    #     :return: None
    #
    #     """
    #     dataset = self.to_dict()
    #
    #     save_path = self.PATH + dataset_name + "_data.json"
    #
    #     with open(save_path, "w") as jfile:
    #         json.dump(dataset, jfile)