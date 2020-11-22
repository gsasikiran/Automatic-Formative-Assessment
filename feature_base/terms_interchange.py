from formative_assessment.utilities.utils import Utilities


class InterchangeOfTerms:
    def __init__(self):
        pass

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
