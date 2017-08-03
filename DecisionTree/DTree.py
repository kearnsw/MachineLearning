from math import log2 as log
from parse_vectors import parse, vectorize
import pickle
import sys
import os.path
from config import TOPICS, TOPIC_LIST


class Node:
    def __init__(self, doc_ids, parent, path):
        self.id = path
        self.docs = doc_ids
        self.parent = parent
        self.entropy = entropy(ratio(doc_ids))
        self.left = None
        self.right = None
        self.num_docs = sum([len(t) for t in doc_ids])
        self.split_criterion = None
        self.max_IG = 0

    def print(self, out_file):
        out_file.write(" ".join([self.id[:-1], str(self.num_docs), ""]))
        for idx, topic in enumerate(self.docs):
            out_file.write(" ".join([TOPIC_LIST[idx], str(len(self.docs[idx])), ""]))
        out_file.write("\n")


class Tree:
    def __init__(self, root, features, documents):
        self.root = root
        self.features = features
        self.docs = documents

    @staticmethod
    def branch(left_node, right_node):
        if left_node:
            parent = left_node.parent
        else:
            parent = right_node.parent
        parent.left = left_node
        parent.right = right_node

    def grow(self, max_depth, min_gain):
        """
        Depth-first tree building

        :return: 0
        """
        out = open(sys.argv[5], "w")
        # Depth-first tree building
        pointer = self.root
        stack = [False]
        while pointer:
            left, right = max_split(tree.features, tree.docs, pointer, min_gain)
            if left or right:
                tree.branch(left, right)        # Add nodes to tree
            else:
                pointer.print(out)              # Print terminal node
            if left and len(left.id.split("&")) <= max_depth:
                stack.append(left)
                stack.append(right)
            pointer = stack.pop()

        return 0

    def save(self, out_file):
        pickle.dump(self, open(out_file, "wb"))

    def classify(self, feature_list, topics):
        """
        Classifies a document based on its feature list

        :param feature_list: list of features present in the document
        :param topics: list of the classes to be determined
        :return: the class of the document
        """
        if self.root.split_criterion in feature_list:
            curr_node = self.root.left
        else:
            curr_node = self.root.right

        prev_node = Node([], None, "Dummy")
        while curr_node is not None:
            prev_node = curr_node
            if curr_node.split_criterion in feature_list:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right

        return prev_node.docs


def max_split(features, documents, parent, min_gain):
    """
    Split on each feature and calculate entropy then determine the

    :param features: list of attributes on which to split, i.e. [w1, w2, w3]
    :param documents: list of documents split by topic, i.e. [[d1,d2,d3], [d5,d7], [d4,d6]]
    :param parent: Node object of that is the source of the destination nodes being calculated
    :param min_gain: minimum gain to split on
    :return: left Node object, right Node object
    """
    topics = len(documents[0])
    left, right = None, None
    for feature_idx, feature in enumerate(features):

        # Calculate ids
        pos_ids = []
        neg_ids = []
        for topic in range(topics):
            pos_ids.append([id for id in documents[feature_idx][topic] if id in parent.docs[topic]])
            neg_ids.append(list(set(parent.docs[topic]) - set(pos_ids[topic])))

        # Create nodes and calculate entropy
        pos_node = Node(pos_ids, parent, parent.id + feature + "&")
        neg_node = Node(neg_ids, parent, parent.id + "!" + feature + "&")

        if pos_node.num_docs > 0 and neg_node.num_docs > 0:
            infoGain = parent.entropy - pos_node.num_docs / parent.num_docs * pos_node.entropy \
                       - neg_node.num_docs / parent.num_docs * neg_node.entropy

            if infoGain > parent.max_IG and infoGain >= min_gain:
                parent.max_IG = infoGain
                parent.split_criterion = features[feature_idx]
                left, right = pos_node, neg_node

    return left, right


def entropy(proportions):
    total = 0
    for p in proportions:
        if p == 0:
            continue
        total -= p * log(p)
    return total


def ratio(lists):
    """
    Calculates the ratio for each topic

    :param lists: list containing a list of documents for each topic, e.g. for 3 classes: [[d1,d2,d3],[d5,d7],[d4,d6]]
    :returns: list of ratios with length equal to number of topics, e.g. [3/7, 2/7, 2/7]

    """
    total = 0
    lengths = []
    for idx, l in enumerate(lists):
        lengths.append(len(l))
        total += lengths[idx]

    ratios = []
    for length in lengths:
        if total == 0:
            ratios.append(0)
        else:
            ratios.append(length/float(total))

    return ratios


def class_argmax(class_list):
    max = 0
    argmax = None
    for idx in range(len(class_list)):
        num_docs = len(class_list[idx])
        if num_docs > max:
            max = num_docs
            argmax = TOPIC_LIST[idx]
    return argmax


def print_classification(class_list, out_file):
    ratios = ratio(class_list)
    for idx, value in enumerate(ratios):
        out_file.write(TOPIC_LIST[idx] + " " + str(value) + "\t")
    out_file.write("\n")


def classify_data(data_file, decision_tree):
    sys_out = open(sys.argv[6], "w")
    print("==============Training===============")
    TP = 0
    FP = 0
    confusion_matrix = [[0 for x in range(len(TOPIC_LIST))] for y in range(len(TOPIC_LIST))]
    counter = 0
    for line in data_file:
        label = line.split()[0]
        vector = vectorize(line)
        classification_list = decision_tree.classify(vector, TOPIC_LIST)

        sys_out.write("array:" + str(counter) + "\t")
        print_classification(classification_list, sys_out)

        classification = class_argmax(classification_list)
        confusion_matrix[TOPICS[classification]][TOPICS[label]] += 1
        if classification == label:
            TP += 1
        else:
            FP += 1
        counter += 1
    sys.stdout.write("Confusion Matrix:\n\t\t\t\t")
    for topic in TOPIC_LIST:
        sys.stdout.write(topic + " ")
    sys.stdout.write("\n")
    for idx, row in enumerate(confusion_matrix):
        sys.stdout.write(TOPIC_LIST[idx] + "\t")
        for element in row:
            sys.stdout.write(str(element) + " ")
        sys.stdout.write("\n")

    print("\nAccuracy: " + str(TP / (TP + FP)) + "\n")

if __name__ == "__main__":

    # Doc_list contains documents for each term
    # Term_list contains the terms for decoding output
    # Class_list contains the split of classes
    train_file = open(sys.argv[1], "r")
    doc_lists, term_list, class_list = parse(train_file)
    train_file.close()
    # Calculate head node entropy
    head = Node(class_list, None, "")
    tree = Tree(head, term_list, doc_lists)
    tree.grow(int(sys.argv[3]), float(sys.argv[4]))

    train_file = open(sys.argv[1], "r")
    test_file = open(sys.argv[2], "r")

    classify_data(train_file, tree)
    classify_data(test_file, tree)

