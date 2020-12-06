from formative_assessment.dataset_extractor import DataExtractor
from formative_assessment.feature_extractor import FeatureExtractor


if __name__ == '__main__':
    PATH = 'dataset/mohler/'

    df = DataExtractor(PATH)
    id = 2.3
    print('Id = ', id)
    question = df.get_questions(id)
    desired_answer = df.get_desired_answers(id)
    print(desired_answer)

    student_answers = df.get_student_answers(id)

    # student_answer = "the main advantages to object-oriented programming  is data abstraction, easier maintenance, " \
    #                  "and re-usability." #df.get_student_answers(id)

    for student_answer in student_answers:

        print(student_answer)
        extract_features = FeatureExtractor(id, student_answer, dataset_path=PATH)
        # extract_features.get_wrong_terms()
        # extract_features.is_wrong_answer() # Wrong answers work only when get_wrong_terms method is run, else returns None
        extract_features.get_interchanged_terms()
        print("----------------------------------------------------------")

