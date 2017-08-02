import sys
from collections import defaultdict, Counter
from math import inf
import pandas as pd
import numpy as np
from math import log10


def log(number):
    if number == 0:
        print("ZOMG ZERO VALUE")
        return 0
    else:
        return log10(number)


class Multinomial_NB:
    def __init__(self, class_delta=0, feat_delta=0):
        self.class_delta = float(class_delta)
        self.feat_delta = float(feat_delta)
        self.corpus = defaultdict(Counter)
        self.class_probs = {}
        self.class_qsum = {}
        self.vocab = []
        self.topic_counter = Counter()
        self.topics = Counter()
        self.topic_dict = {}

    def train(self, input_file):
        f = open(input_file, "r")
        for doc in f:
            topic, word_counts = self.vectorize(doc)
            self.topics[topic] += 1
            for word, count in word_counts:
                self.corpus[topic][word] += int(count)
                self.topic_counter[topic] += int(count)
                self.vocab.append(word)
        f.close()
        self.vocab = list(set(self.vocab))

        for topic in self.corpus:
            for word in self.vocab:
                self.corpus[topic][word] = (self.feat_delta + self.corpus[topic][word]) / \
                                           (self.feat_delta * len(self.vocab) + self.topic_counter[topic])
        # Calculate p(c)
        for topic, count in self.topics.items():
            self.class_probs[topic] = (self.class_delta + count) / (2 * self.class_delta + sum(self.topics.values()))

        for idx, topic in enumerate(self.topic_counter):
            self.topic_dict[topic] = idx

        return self.corpus

    def classify(self, word_counts):
        probabilities = []
        max_prob = -inf
        argmax = None
        for c in self.corpus:
            feat_sum = 0
            for word, count in word_counts:
                if self.corpus[c][word] == 0:
                    continue
                feat_sum += int(count) * log(self.corpus[c][word])

            prob = log(self.class_probs[c]) + feat_sum
            if prob > max_prob:  # Check if prob is higher and replace argmax
                argmax = c
                max_prob = prob
            probabilities.append((prob, c))

        probabilities.sort(key=lambda x: x[0], reverse=True)

        return argmax, probabilities

    def test(self, input_file):
        num_correct = 0
        doc_num = 0
        output = []
        confusion_matrix = np.zeros((len(self.topic_counter), len(self.topic_counter)))

        f = open(input_file, "r")
        for doc in f:
            true_topic, word_counts = self.vectorize(doc)
            argmax, predictions = self.classify(word_counts)
            if argmax == true_topic:
                num_correct += 1
            output.append("array" + str(doc_num) + ": " + argmax + " ")
            confusion_matrix[self.topic_dict[true_topic]][self.topic_dict[argmax]] += 1

            # Convert probs
            max_prob = predictions[0][0]
            probabilities = [pow(10, prob / 10 - max_prob / 10) for prob, topic in predictions]
            total = sum(probabilities)
            for idx, prob in enumerate(probabilities):
                topic = predictions[idx][1]
                output.append(topic + " " + str(prob / total) + " ")
            output.append("\n")
            doc_num += 1
        f.close()

        # Print Confusion Matrix as a Table
        df = pd.DataFrame(confusion_matrix, columns=self.topic_counter.keys(), index=self.topic_counter.keys())
        pd.set_option('expand_frame_repr', False)
        print("Confusion matrix for the training data: (row is the truth, column is the system output)\n")
        print(df)

        # Print Accuracy
        accuracy = num_correct / doc_num
        print(input_file + " Accuracy: " + str(accuracy) + "\n")
        return output

    @staticmethod
    def vectorize(doc):
        """
        Transform a document to a vector array of binary features.
        :param doc: document in MALLET SVMlight format
        :return: an array of length |V|, where |V| is the vocabulary size.
        """
        doc = doc.split()
        topic = doc[0]
        word_counts = [wc.split(":") for wc in doc[1:]]

        return topic, word_counts

    def print(self, model_file):
        """
        Prints the prior probabilities of the model file with first section containing class prior probabilities and the
        following sections contain the probabilities for each feature given the class p(f|c)
        :param model_file: the file name to output the model
        :return: None
        """
        out_file = open(model_file, "w")

        out_file.write("%%%%% prior prob P(c) %%%%%\n")
        for topic, prob in self.class_probs.items():
            out_file.write(topic + "\t" + str(prob) + "\t" + str(log(prob)) + "\n")

        out_file.write("%%%%% conditional prob P(f|c) %%%%%\n")
        for topic, word_counts in self.corpus.items():
            out_file.write("%%%%% conditional prob P(f|c) c=" + topic + " %%%%%\n")
            sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            for word, prob in sorted_word_counts:
                out_file.write("\t".join([word, topic, str(prob), str(log(prob))]) + "\n")

        out_file.close()
        return None


if __name__ == "__main__":

    train_data = sys.argv[1]
    test_data = sys.argv[2]
    class_prior_delta = sys.argv[3]
    cond_prob_delta = sys.argv[4]

    mnb = Multinomial_NB(class_delta=class_prior_delta, feat_delta=cond_prob_delta)
    mnb.train(train_data)
    mnb.print(sys.argv[5])

    sys_output = open(sys.argv[6], "w")
    sys_output.write("%%%%% training data:\n")
    train_output = mnb.test(train_data)
    for line in train_output:
        sys_output.write(line)
    sys_output.write("%%%%% testing data:\n")
    test_output = mnb.test(test_data)
    for line in test_output:
        sys_output.write(line)
    sys_output.close()
