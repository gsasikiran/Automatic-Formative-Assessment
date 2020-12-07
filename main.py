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

__author__ = "Sasi Kiran Gaddipati"
__credits__ = []
__license__ = ""
__version__ = ""
__last_modified__ = "06.12.2020"
__status__ = "Development"

if __name__ == '__main__':
    PATH = "dataset/mohler/"
    # Convert the data into  dictionary with ids, their corresponding questions, desired answers and student answers
    convert_data = ConvertDataType(PATH)
    dataset_dict = convert_data.to_dict()

    id_list = dataset_dict.keys()

    for s_no in id_list:
        print('Id = ', s_no)

        question = dataset_dict[s_no]["question"]
        print("Question: ", question)

        desired_answer = dataset_dict[s_no]["des_answer"]
        print("Desired answer: ", desired_answer)
        # temporary student answer
        student_answers = dataset_dict[s_no]["stu_answers"]

        # for student_answer in student_answers:
        for answer in student_answers:
            start = time.time()
            print(answer)
            extract_features = FeatureExtractor(s_no, answer, dataset_dict, PATH)
            extract_features.get_wrong_terms()
            extract_features.is_wrong_answer() # Wrong answers work only when get_wrong_terms method is run, else returns None
            extract_features.get_suboptimal_answers()
            # extract_features.get_interchanged_terms()
            print("It took ",  time.time() - start, " secs")
            print("----------------------------------------------------------")
