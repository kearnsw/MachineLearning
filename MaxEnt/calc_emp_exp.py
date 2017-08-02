from maxent_classify import Document
from collections import defaultdict, Counter
import sys


class Corpus:
    def __init__(self):
        self.docs = []
        self.num_docs = 0
        self.topics = None
        self.expectation = defaultdict(Counter)
        self.raw_counts = defaultdict(Counter)

    def load(self, input_file):
        f = open(input_file, "r")

        for line in f:
            doc = Document(text=line)
            self.docs.append(doc.vectorize())

        self.num_docs = len(self.docs)
        self.topics = sorted(set([doc.topic for doc in self.docs]))

    def calc_emp_exp(self):
        for doc in self.docs:
            for word in doc.word_counts:
                self.expectation[doc.topic][word] += 1 / float(self.num_docs)
                self.raw_counts[doc.topic][word] += 1

    def print_expectation(self, output_file):
        out = open(output_file, "w")
        for topic, feat_prob in sorted(self.expectation.items()):
            for feat, prob in sorted(feat_prob.items()):
                out.write(" ".join([topic, feat, str(prob), str(self.raw_counts[topic][feat])]) + "\n")
        out.close()

    def print_model_expectation(self, output_file):
        out = open(output_file, "w")
        for topic, feat_prob in sorted(self.expectation.items()):
            for feat, prob in sorted(feat_prob.items()):
                out.write(" ".join([topic, feat, str(prob), str(prob * self.num_docs)]) + "\n")
        out.close()

if __name__ == "__main__":
    cp = Corpus()
    cp.load(sys.argv[1])
    cp.calc_emp_exp()
    cp.print_expectation(sys.argv[2])
