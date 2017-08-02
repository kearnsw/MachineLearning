import sys
from collections import defaultdict
from math import pow, e
import numpy as np
import pandas as pd

class Document:
    def __init__(self, text):
        self.raw_text = text
        self.topic = None
        self.word_counts = defaultdict(int)
        self.words = None

    def vectorize(self):
        tokens = self.raw_text.split()
        self.topic = tokens[0]
        for word_count in tokens[1:]:
            word, count = word_count.split(":")
            self.word_counts[word] = count
        return self


class MaxEnt:
    def __init__(self):
        self.model = defaultdict(defaultdict)

    def load(self, input_file):
        """
        Load MALLET classifier info file and modify the MaxEnt object attributes.
        :param input_file: MALLET classifier info file
        :return: Model as a dict of dicts, i.e. model[topic][token] = prob
        """
        topic = None
        f = open(input_file, "r")

        for line in f:
            line = line.split()
            if len(line) != 2:
                topic = " ".join(line[3:])
            else:
                token, prob = line
                self.model[topic][token] = float(prob)

        return self.model

    def classify(self, doc):
        topic_probs = []
        total_prob = 0
        for topic, probs in self.model.items():
            total = 0
            for word in doc.word_counts.keys():
                total += self.model[topic][word]
            conditional_prob = pow(e, probs["<default>"] + total)
            topic_probs.append((topic, conditional_prob))
            total_prob += conditional_prob

        return [(topic, prob/total_prob) for topic, prob in topic_probs]

    def test(self, input_file, output_file):
        metrics = {}

        topic_dict = {}
        for idx, topic in enumerate(self.model.keys()):
            topic_dict[topic] = idx

        confusion_matrix = np.zeros((len(self.model), len(self.model)))

        true = 0
        total = 0

        f = open(input_file, "r")
        out = open(output_file, "w")
        for line in f:
            doc = Document(text=line).vectorize()
            results_ = self.classify(doc)
            top_results = sorted(results_, key=lambda x: x[1], reverse=True)
            top_result = top_results[0]

            i = topic_dict[doc.topic]
            j = topic_dict[top_result[0]]
            confusion_matrix[i][j] += 1
	    
            if doc.topic == top_result[0]:
                true += 1
            out.write("array" + str(total + 1) + ": " + top_result[0] + " ")
            for topic, prob in top_results:
                out.write(topic + " " + str(prob) + " ")
            out.write("\n")
            total += 1

        metrics["accuracy"] = true/total

        # Print Confusion Matrix as a Table
        df = pd.DataFrame(confusion_matrix, columns=sorted(topic_dict.keys()), index=sorted(topic_dict.keys()))
        pd.set_option('expand_frame_repr', False)
        print("Confusion matrix: (row is the truth, column is the system output)\n")
        print(df)
        print("Accuracy: " + str(metrics["accuracy"]))
        return metrics


if __name__ == "__main__":

    test_data = sys.argv[1]
    model_file = sys.argv[2]
    sys_output = sys.argv[3]

    clf = MaxEnt()
    clf.load(model_file)
    results = clf.test(test_data, sys_output)
