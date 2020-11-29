from formative_assessment.utilities.utils import Utilities

class InterchangeOfTerms:
    def __init__(self):
        self.utils = Utilities()

    @staticmethod
    def get_question_terms(question: str):
        """
            Returns the key terms from the question
        :param question: str
        :return: List[str]
            Return the list of strings of key terms
        """
        utils = Utilities()
        return utils.extract_phrases_tr(question)

    def get_head(self, question: str, des_ans: str):

        ques_len = len(question)

        combined_str = question + " " + des_ans
        resolved_str = self.utils.corefer_resolution(combined_str)

        # Ignoring the space between the question and desired answer
        des_ans = resolved_str[ques_len + 1: ]

        return self.utils.get_common_keyphrases(question, des_ans)


