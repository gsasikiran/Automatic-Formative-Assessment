import re
import numpy as np

from formative_assessment.utilities.utils import Utilities
from formative_assessment.utilities.embed import AssignEmbedding


class InterchangeOfTerms:
    def __init__(self):
        self.utils = Utilities.instance()
        self.embed = AssignEmbedding("fasttext")

    def coref_res(self, question, desired_answer, student_answer):

        ques_len: int = len(question)
        resolved_ques = self.utils.corefer_resolution(question)

        combined_des_ans: str = resolved_ques + " " + desired_answer
        combined_stu_ans: str = resolved_ques + " " + student_answer

        resolved_des_ans: str = self.utils.corefer_resolution(combined_des_ans)
        resolved_stu_ans: str = self.utils.corefer_resolution(combined_stu_ans)

        return resolved_ques, resolved_des_ans[ques_len + 1:], resolved_stu_ans[ques_len + 1:]

    def get_question_terms(self, question: str):
        """
            Returns the key terms from the question
        :param question: str
        :return: List[str]
            Return the list of strings of key terms
        """
        return self.utils.extract_phrases_rake(question)

    # def _get_common_keyphrases(self, question, answer):
    #
    #     # text1_kp = set(self.extract_phrases_tr(text1))
    #     # text2_kp = set(self.extract_phrases_tr(text2))
    #     text1_kp = set(self.utils.extract_phrases_rake(question))
    #     text2_kp = set(self.utils.extract_phrases_rake(answer))
    #
    #     text1_updated = set()
    #     text2_updated = set()
    #
    #     for text in text1_kp:
    #         filtered_text = self.remove_articles(text)
    #         lemmas = self.lemmatize(filtered_text)
    #         filtered_text = " ".join(lemmas)
    #         text1_updated.add(filtered_text)
    #
    #     for text2 in text2_kp:
    #         filtered_text = self.remove_articles(text)
    #         lemmas = self.lemmatize(filtered_text)
    #         filtered_text = " ".join(lemmas)
    #         text2_updated.add(filtered_text)
    #
    #     return text1_updated.intersection(text2_updated)

    def get_topics(self, question: str, des_ans: str):
        """
            Generate the head words of which the question is asked about.
        :param question: str
        :param des_ans: str
        :return: set
            Returns the set of head words
        """


        # Co-reference resolution for the combination of question and desired answer


        # Extracting desired answer from co-reference resolution
        # Ignoring the space between the question and desired answer

        # Text rank key phrase extraction
        # question_kp = self.utils.extract_phrases_tr(question)
        # question_kp = self.utils.extract_phrases_rake(resolved_ques)

        # Generate the common key phrases between the question and desired answer
        common_kp = self.utils.get_common_keyphrases(question, des_ans)

        # TODO: Consider extracting the heads from student answers, if the desired answer did not result in any of the common words

        # Return the first extracted key word of the question, if no common words are extracted
        # if len(common_kp) == 0:
        #     return {self.utils.remove_articles(question_kp[0])} if len(question_kp) > 0 else []

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

                sent = ""

                for str in relation:
                    if re.match("V: ", str):
                        verb = re.sub("V: ", '', str)
                        lemmas = self.utils.lemmatize(verb)
                        verb = " ".join(lemmas)
                        sent = sent + " " + verb
                        continue

                    elif re.match("ARG(\d|M(-TMP|-LOC|-NEG))", str):
                        text = re.sub("ARG(\d|M(-TMP|-LOC|-NEG)): ", '', str)
                        lemmas = self.utils.lemmatize(text)
                        arg = " ".join(lemmas)
                        if topic in arg:
                            head = topic
                            continue
                        else:
                            args.append(arg)
                            sent = sent + " " + arg
                        continue

                if topic == head:
                    tree[topic].append(sent[1:])
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

    def is_interchanged(self, des_tree, stu_tree, interchange_threshold=0.5):

        des_sents = des_tree  # self._create_sent_tree(des_tree)
        stu_sents = stu_tree  # self._create_sent_tree(stu_tree)

        des_values = list(des_sents.values())
        des_values = [self.utils.remove_articles(item) for sublist in des_values for item in sublist]

        missed_topics = set()
        interchanged = []

        print("Interchange of phrases: ")
        if des_values:
            des_embeds = self.embed.assign(des_values)

            for topic in stu_sents:
                if des_sents[topic] and not stu_sents[topic]:
                    print("You have not answered about the topic \'" + topic + "\'")
                    missed_topics.add(topic)

                else:
                    # TODO: can create a similarity matrix instead of two for loops
                    for sent in stu_sents[topic]:

                        sent = self.utils.remove_articles(sent)
                        sent_embed = self.embed.assign([sent])  # [0]
                        sim_scores = []

                        for embed in des_embeds:
                            sim_scores.append(self.utils.get_cosine_similarity(sent_embed, embed))

                        if np.max(np.asarray(sim_scores)) > interchange_threshold:
                            index = int(np.argmax(np.asarray(sim_scores)))

                            if des_values[index] in des_sents[topic]:
                                # print("The sentence \"" + sent + "\" is correctly written for topic \"" + topic +
                                # "\"")
                                continue
                            else:
                                for des_topic in des_sents:
                                    if des_values[index] in des_sents[des_topic]:

                                        print("You have interchanged the terms/phrase of topic \"" + des_topic + "\" to "
                                              "the topic \"" + topic + "\" for the sentence \"" + sent + "\"")
                                        interchanged.append([des_topic, topic, sent])

                        else:
                            continue

        return interchanged, missed_topics
