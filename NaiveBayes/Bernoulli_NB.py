import sys
from collections import defaultdict, Counter
from math import inf, log10
import pandas as pd
import numpy as np


def log(number):
    if number == 0:
        return 0
    else:
        return log10(number)


class Bernoulli_NB:

    def __init__(self, class_delta=0, feat_delta=0):
        self.class_delta = float(class_delta)
        self.feat_delta = float(feat_delta)
        self.corpus = defaultdict(Counter)
        self.class_probs = {}
        self.class_qsum = Counter()
        self.vocab = []
        self.topic_counter = Counter()
        self.topic_dict = {}

    def train(self, input_file):
        """
        Train the model by calculating prior probabilities using a binary feature method
        :param input_file:
        :return:
        """
        # read data
        f = open(input_file, "r")
        for document in f:
            topic, words = self.vectorize(document)
            self.topic_counter[topic] += 1
            for word in words:
                self.corpus[topic][word] += 1
                self.vocab.append(word)
        self.vocab = list(set(self.vocab))
        f.close()

        # Calculate p(f|c) with Smoothing
        # Calculate log probabilities for each word in each class
        for word in self.vocab:
            for topic in self.corpus:
                smoothed = (self.feat_delta + self.corpus[topic][word]) / (2 * self.feat_delta + self.topic_counter[topic])
                self.corpus[topic][word] = smoothed
                self.class_qsum[topic] += log(1 - smoothed)

        # Calculate p(c)
        for topic, count in self.topic_counter.items():
            self.class_probs[topic] = (self.class_delta + count) / (2 * self.class_delta + sum(self.topic_counter.values()))

        for idx, topic in enumerate(self.topic_counter):
            self.topic_dict[topic] = idx

        return self.corpus

    def classify(self, bag):
        """
        Determines the argmax of classes from prior probabilities learned during training
        :param bag: tokens to be used in classifier
        :return: argmax of classes
        """
        max_prob = -inf
        argmax = None
        probabilities = []
        for c in self.corpus:                   # Determine the probability of each class
            feat_sum = 0
            for t in bag:
                if self.corpus[c][t] == 0:
                    continue
                feat_prob = self.corpus[c][t]
                feat_sum += log(feat_prob) - log(1-feat_prob)
            prob = feat_sum + log(self.class_probs[c]) + self.class_qsum[c]
            probabilities.append((prob, c))

            if prob > max_prob:                 # Check if prob is higher and replace argmax
                argmax = c
                max_prob = prob
        probabilities.sort(key=lambda x: x[0], reverse=True)
        return argmax, probabilities

    @staticmethod
    def vectorize(doc):
        """
        Transform a document to a vector array of binary features.
        :param doc: document in MALLET SVMlight format
        :return: an array of length |V|, where |V| is the vocabulary size.
        """
        doc = doc.split()
        topic = doc[0]
        words = [wc.split(":")[0] for wc in doc[1:]]

        return topic, words

    def test(self, input_file):
        """
        Classify all documents in an input file and report performance metrics
        :param input_file: file to be classified in MALLET SVMlight format
        :return: performance metrics
        """

        num_correct = 0
        doc_num = 0
        output = []
        confusion_matrix = np.zeros((len(self.topic_counter), len(self.topic_counter)))

        f = open(input_file, "r")
        for doc in f:
            true_topic, bag_of_words = self.vectorize(doc)
            argmax, predictions = self.classify(bag_of_words)
            if argmax == true_topic:
                num_correct += 1
            output.append("array" + str(doc_num) + ": " + argmax + " ")
            confusion_matrix[self.topic_dict[true_topic]][self.topic_dict[argmax]] += 1
            # Convert probs
            max_prob = predictions[0][0]
            probabilities = [pow(10, prob - max_prob) for prob, topic in predictions]
            total = sum(probabilities)
            for idx, prob in enumerate(probabilities):
                topic = predictions[idx][1]
                output.append(topic + " " + str(prob/total) + " ")
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

    bnb = Bernoulli_NB(class_delta=class_prior_delta, feat_delta=cond_prob_delta)
    bnb.train(train_data)

    # Print model
    bnb.print(sys.argv[5])

    # Print Classification Result
    sys_output = open(sys.argv[6], "w")
    sys_output.write("%%%%% training data:\n")
    train_output = bnb.test(train_data)
    for line in train_output:
        sys_output.write(line)
    sys_output.write("%%%%% testing data:\n")
    test_output = bnb.test(test_data)
    for line in test_output:
        sys_output.write(line)
    sys_output.close()
