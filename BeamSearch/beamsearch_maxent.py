import sys
from math import exp, log10
from queue import Queue
from maxent_classify import MaxEnt


class Node:
    def __init__(self, parent, tag, prob, word_prob):
        self.parent = parent
        self.tag = tag
        self.probability = prob
        self.word_prob = word_prob

    def spawn(self, tag, probability):
        return Node(parent=self, tag=tag, prob=(probability * self.probability), word_prob=probability)


class Beam_Tree:
    def __init__(self, root, n, k, size, model):
        self.root = root
        self.top_K = []
        self.top_N = []
        self.model = model
        self.n = n
        self.k = k
        self.beam_size = size
        self.true_seq = []

    def grow(self, data, num_tokens):
        """
        Begins with root node and grows a tree by first spawning the top N nodes with highest entropy then pruning over
        the iteration and adding the top K nodes to the next iteration, until all tokens have been exhausted.

        :param data: feature vectors in Mallet MaxEnt model format
        :param num_tokens: number of tokens for the current sentence
        :return: Nothing, generates a tree pruned using beam search method
        """
        q = Queue()
        q.put(self.root)
        self.true_seq = []
        for i in range(num_tokens):
            features = data.readline().strip().split()
            self.true_seq.append((features[0], features[1]))
            self.top_K = []
            # Loop through all nodes in the top K nodes
            while not q.empty():
                curr_node = q.get()
                self.top_N = []
                self.spawn_top_N(curr_node, features[2:])

                for node in self.top_N:
                    self.top_K.append(node)

            # Add top nodes to next iteration's
            for node in self.prune():
                q.put(node)

        self.top_K = sorted(self.top_K, key=lambda x: x.probability, reverse=True)[:self.k]

    def spawn_top_N(self, node, features):
        """
        Create N child nodes for a given node that correspond to the highest probability

        :param node: Node from which to spawn child nodes
        :param features: feature vector from which to calculate child probabilities
        :return: the top N children of the node
        """
        # Get the previous and previous two tags
        prev_tag = "prevT=" + node.tag
        prev_2_tags = "prevTwoTags=" + node.parent.tag + "+" + node.tag

        # Calculate the entropy of each tag from the current node
        entropies = []
        for tag in self.model.keys():
            entropies.append((tag, calc_entropy(tag, features, prev_tag, prev_2_tags, self.model)))
        entropies = sorted(entropies, key=lambda x: x[1], reverse=True)

        # normalize entropy by dividing numerators by the total for all numerators
        for tag, ent in entropies[:self.n]:
            new_node = node.spawn(tag, ent / sum(e[1] for e in entropies))
            self.top_N.append(new_node)

    def prune(self):
        self.top_K = sorted(self.top_K, key=lambda x: x.probability, reverse=True)
        max_prob = max(node.probability for node in self.top_K)
        return [node for node in self.top_K[:self.k] if (log10(node.probability) + self.beam_size) >= log10(max_prob)]

    def backtrace(self):
        node = self.top_K[0]
        output = [(node.tag, node.word_prob)]
        while node.parent != self.root:
            output.append((node.tag, node.word_prob))
            node = node.parent
        output.append((node.tag, node.word_prob))
        return output[::-1]

def beam_search(data, boundary, model, n, k, size):
    root = Node(Node(None, "BOS", 1.0, 1.0), "BOS", 1.0, 1.0)
    tree = Beam_Tree(root, n, k, size, model)
    correct = 0
    total = 0
    for line_num, line in enumerate(boundary):
        if line_num % 100 == 0:
            sys.stderr.write("Processing sentence " + str(line_num) + "\n")
            sys.stderr.flush()
        tree.grow(data, num_tokens=int(line.strip()))
        backtrace = tree.backtrace()
        for idx, truth in enumerate(tree.true_seq):
            sys.stdout.write(truth[0] + " " + truth[1] + " " + backtrace[idx][0] + " " + str(backtrace[idx][1]) + "\n")
            if truth[1] == backtrace[idx][0]:
                correct += 1
            total += 1
    sys.stderr.write("Accuracy: " + str(correct/total) + "\n")

def calc_entropy(tag, features, prev_tag, prev2_tags, model):
    total = model[tag]["<default>"]
    for feature in features:
        if feature != "1":
            total += clf.model[tag][feature]
    if prev2_tags in model[tag].keys():
        total += clf.model[tag][prev_tag] + clf.model[tag][prev2_tags]
    else:
        total += clf.model[tag][prev_tag]
    return exp(total)

if __name__ == "__main__":
    test_data = sys.argv[1]
    boundary_fn = sys.argv[2]
    model_fn = sys.argv[3]
    beam_size = float(sys.argv[4])
    N = int(sys.argv[5])
    K = int(sys.argv[6])
    clf = MaxEnt()
    sys.stderr.write("Loading model...\n")
    clf.load(model_fn)

    input_file = open(test_data, "r")
    boundary_file = open(boundary_fn, "r")
    sys.stderr.write("Performing beam search...\n")
    beam_search(input_file, boundary_file, clf.model, n=N, k=K, size=beam_size)

    input_file.close()
    boundary_file.close()
