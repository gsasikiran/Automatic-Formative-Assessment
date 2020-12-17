import re
import numpy as np

from formative_assessment.utilities.utils import Utilities, PreProcess
from formative_assessment.utilities.embed import Embedding

class InterchangeOfTerms:
    def __init__(self):
        self.utils = Utilities()
        self.preprocess = PreProcess()
        self.embed = Embedding()

    def get_question_terms(self, question: str):
        """
            Returns the key terms from the question
        :param question: str
        :return: List[str]
            Return the list of strings of key terms
        """
        return self.utils.extract_phrases_tr(question)

    def get_topics(self, question: str, des_ans: str):
        """
            Generate the head words of which the question is asked about.
        :param question: str
        :param des_ans: str
        :return: set
            Returns the set of head words
        """
        ques_len: int = len(question)

        # Co-reference resolution for the combination of question and desired answer
        resolved_ques = self.utils.corefer_resolution(question)
        combined_str: str = resolved_ques + " " + des_ans
        resolved_str: str = self.utils.corefer_resolution(combined_str)

        # Ignoring the space between the question and desired answer
        des_ans = resolved_str[ques_len + 1:]

        # Text rank key phrase extraction
        question_kp = self.utils.extract_phrases_tr(resolved_ques)

        # Generate the common key phrases between the question and desired answer
        common_kp = self.utils.get_common_keyphrases(question, des_ans)

        # TODO: We can extract phrases such that if the text rank phrases are in the flair phrases.
        # TODO: Also if the returned set do not have the first text rank key phrase of the question, add that to the set.
        # TODO: Try to extract key phrases by considering the text rank key phrases weightage.
        # TODO: Consider extracting the heads from student answers, if the desired answer did not result in any of the common words

        # Return the first extracted key word of the question, if no common words are extracted
        if len(common_kp) == 0:
            return {self.utils.remove_articles(question_kp[0])} if len(question_kp) > 0 else []

        return common_kp

    def generate_tree(self, heads: str, answer: str):

        # TODO: check the relation extraction for the sentences with verbs with "be" form.
        #  For example, "Constructor is a first method in a class."

        relations = self.utils.open_ie(answer)

        tree = {}

        for topic in heads:
            tree[topic] = []

            for relation in relations:

                # relation_dict[topic] = []
                args = []
                verb = ""
                head = ""

                for str in relation:
                    if re.match("V: ", str):
                        verb = re.sub("V: ", '', str)
                        lemmas = self.preprocess.lemmatize(verb)
                        verb = self.utils.tokens_to_str(lemmas)
                        continue

                    elif re.match("ARG(\d|M(-TMP|-LOC))", str):
                        text = re.sub("ARG(\d|M(-TMP|-LOC)): ", '', str)
                        lemmas = self.preprocess.lemmatize(text)
                        arg = self.utils.tokens_to_str(lemmas)
                        if topic in arg:
                            head = topic
                            continue
                        else:
                            args.append(arg)
                        continue

                if topic == head:
                    for arg in args:
                        tree[topic].append((verb, arg))
        return tree

    @staticmethod
    def _create_sent_tree(tree):

        sents = {}
        for topic in tree:
            sents[topic] = []
            for att in tree[topic]:
                sent = att[0] + " " + att[1]
                sents[topic].append(sent)

        return sents

    def is_interchanged(self, des_tree, stu_tree):

        des_sents = self._create_sent_tree(des_tree)
        stu_sents = self._create_sent_tree(stu_tree)

        print(des_tree, stu_tree)

        des_values = list(des_sents.values())
        des_values = [item for sublist in des_values for item in sublist]

        if des_values:
            des_embeds = self.embed.use(des_values)

            for topic in stu_sents:

                for sent in stu_sents[topic]:
                    sent_embed = self.embed.use([sent])[0]
                    sim_scores = []

                    for embed in des_embeds:
                        sim_scores.append(self.utils.get_cosine_similarity(sent_embed, embed))

                    index = int(np.argmax(np.asarray(sim_scores)))

                    if des_values[index] in des_sents[topic]:
                        continue
                    else:
                        for des_topic in des_sents:
                            if des_values[index] in des_sents[des_topic]:
                                print("You have interchanged the terms/phrase of topic \"" + des_topic + "\" to the topic \"" + topic, "\" for the sentence \"" + sent + "\"")