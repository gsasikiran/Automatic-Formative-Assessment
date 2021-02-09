"""
Run the formative assessment on the student answers for a given question and desired answer
Prints the
    Wrong terms/phrases in the student answers if any
    Wrong answer if any
    Interchange of terms/definitions if any

Input: Directory path of the dataset. The directory path should contain answers.csv, questions.csv files
    The questions.csv should consist of ids of the questions, questions and corresponding desired answers
       | id | question | solution |
    The answers.csv should consist of the student answers, their corresponding ids
        |id | answer |
"""
from formative_assessment.dataset_extractor import DataExtractor, ConvertDataType
from formative_assessment.feature_extractor import FeatureExtractor
import time
import warnings
import random

warnings.filterwarnings("ignore")

__author__ = "Sasi Kiran Gaddipati"
__credits__ = []
__license__ = ""
__version__ = ""
__last_modified__ = "08.02.2020"
__status__ = "Development"

if __name__ == '__main__':
    PATH = "dataset/mohler/"

    # Convert the data into  dictionary with ids, their corresponding questions, desired answers and student answers
    convert_data = ConvertDataType(PATH)
    dataset_dict = convert_data.to_dict()

    id_list = list(dataset_dict.keys())

    r = random.Random(5)

    for i in range(0,20):

        s_no = r.choice(id_list)
        index = r.randint(0, 10)
        print('Id = ', s_no)

        question = dataset_dict[s_no]["question"]
        print("Question: ", question)

        desired_answer = dataset_dict[s_no]["des_answer"]
        print("Desired answer: ", desired_answer)
        # temporary student answer
        student_answers = dataset_dict[s_no]["stu_answers"]
        scores = dataset_dict[s_no]["scores"]

        # for i, _ in enumerate(student_answers):
        start = time.time()
        student_answer = student_answers[index]
        print("Student answer: ", student_answer)
        print("Assigned score: ", scores[index])
        extract_features = FeatureExtractor(s_no, student_answer, dataset_dict, PATH)
        extract_features.get_incorrect_terms()
        extract_features.is_wrong_answer()
        # Wrong answers work only when get_wrong_terms method is run, else returns None
        extract_features.get_partial_answers()
        extract_features.get_interchanged_terms()
        print("It took ", time.time() - start, " secs")
        print("----------------------------------------------------------")