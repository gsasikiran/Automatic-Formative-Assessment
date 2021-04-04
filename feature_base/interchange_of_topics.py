"""
 Class extracts the interchange of topics and missed topics from student answer
"""
import re
import numpy as np
from typing import List

from formative_assessment.utilities.utils import Utilities
from formative_assessment.utilities.embed import AssignEmbedding

__author__ = "Sasi Kiran Gaddipati"
__credits__ = ["Tim Metzler"]
__license__ = ""
__version__ = "1.0.1"
__email__ = "sasi-kiran.gaddipati@smail.inf.h-brs.de"
__last_modified__ = "04.04.2021"
__status__ = "Prototype"



class InterchangeOfTopics:
    def __init__(self):
        self.utils = Utilities.instance()
        self.embed = AssignEmbedding("fasttext")

    def _get_common_keyphrases(self, text1, text2):
        """
            Return the common normalized keyphrases between text1 and text2

        :param text1: str
        :param text2: st

        :return: set
            Common keyphrases in both the texts
        """

        # Extract text rank keyphrases as they provide only noun based keyphrases
        text1_kp = set(self.utils.extract_phrases_tr(text1))
        text2_kp = set(self.utils.extract_phrases_tr(text2))
        # text1_kp = set(self.extract_phrases_rake(text1))
        # text2_kp = set(self.extract_phrases_rake(text2))

        text1_updated = set()
        text2_updated = set()

        # Normalizing the keyphrases
        for text in text1_kp:
            filtered_text = self.utils.remove_articles(text)
            # we extract the lemmas, as we focus on root word for checking the common terms.
            lemmas = self.utils.lemmatize(filtered_text)
            filtered_text = " ".join(lemmas)
            text1_updated.add(filtered_text)

        for text in text2_kp:
            filtered_text = self.utils.remove_articles(text)
            lemmas = self.utils.lemmatize(filtered_text)
            filtered_text = " ".join(lemmas)
            text2_updated.add(filtered_text)

        return text1_updated.intersection(text2_updated)

    def get_topics(self, question: str, des_ans: str):
        """
            Generate the topics of which the question is asked about.
        :param question: str
        :param des_ans: str
        :return: set
            Returns the set of topics
        """

        # Generate the common key phrases between the question and desired answer
        topics = self._get_common_keyphrases(question, des_ans)

        return topics

    def generate_tree(self, heads: List[str], answer: str):
        """
          Create a tree with topics as heads and relations as branches.

        :param heads: List[str]
        :param answer: str
        :return: List[str]
        """

        relations = self.utils.relation_extraction(answer)

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

    def get_interchanged_and_missed(self, des_tree: dict, stu_tree: dict, interchange_threshold: float = 0.5):
        """
            From the input, desired and student trees, extracts if any interchanged and missed topics

        :param des_tree: Dict
        :param stu_tree: Dict
        :param interchange_threshold: float
        :return:
            interchanged topics: List
            Missed topics: set
        """

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
                                        print(
                                            "You have interchanged the terms/phrase of topic \"" + des_topic + "\" to "
                                                                                                               "the topic \"" + topic + "\" for the sentence \"" + sent + "\"")
                                        interchanged.append([des_topic, topic, sent])

                        else:
                            continue

        return interchanged, missed_topics

    # @staticmethod
    # def _create_sent_tree(tree):
    #
    #     sents = {}
    #     for topic in tree:
    #         sents[topic] = []
    #         for att in tree[topic]:
    #             sent = att[0] + " " + att[1]
    #             sents[topic].append(sent)
    #
    #     return sents
