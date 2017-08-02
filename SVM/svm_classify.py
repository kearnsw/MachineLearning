import sys
from math import exp, tanh

class SV:
    def __init__(self, weight, vector):
        self.weight = weight
        self.vector = parse_bvector(vector)


class SVM:
    def __init__(self, kernel_type=None):
        """
        SVM classifier, implements the decoding of a libSVM model.

        :param kernel_type: linear, polynomial, RBF, sigmoid
        """
        self.kernel_type = kernel_type
        self.gamma = 0
        self.coef = 0
        self.degree = 0
        self.num_classes = 0
        self.labels = None
        self.total_sv = 0
        self.rho = 0
        self.sv = []

    def load(self, model_file):
        with open(model_file, "r") as f:
            header = True
            for ln in f:
                if header:
                    if "kernel_type" in ln:
                        self.kernel_type = ln.split()[1]
                    elif "degree" in ln:
                        self.degree = float(ln.split()[1])
                    elif "gamma" in ln:
                        self.gamma = float(ln.split()[1])
                    elif "coef0" in ln:
                        self.coef = float(ln.split()[1])
                    elif "rho" in ln:
                        self.rho = float(ln.split()[1])
                    elif "label" in ln:
                        self.labels = ln.split()[1:]
                    elif "SV" in ln:
                        header = False
                else:
                    ln = ln.strip().split()
                    self.sv.append(SV(float(ln[0]), ln[1:]))
        return self

    def classify(self, vector):
        total = -self.rho
        for sv in self.sv:
            total += sv.weight * self.kernel_function(sv.vector, vector)
        if total >= 0:
            return 0, total
        else:
            return 1, total

    def kernel_function(self, v1, v2):
        if self.kernel_type == "linear":
            return dot_product(v1, v2)
        elif self.kernel_type == "polynomial":
            return pow((self.gamma * dot_product(v1, v2) + self.coef), self.degree)
        elif self.kernel_type == "rbf":
            return exp(-self.gamma * squared_euclidean_dist(v1, v2))
        elif self.kernel_type == "sigmoid":
            return tanh(self.gamma * dot_product(v1, v2) + self.coef)
        else:
            sys.stderr.write(self.kernel_type + " kernel type not supported.")


def parse_bvector(vector):
    return [int(x.split(":")[0]) for x in vector]


def dot_product(v1, v2):
    """
    Computes the dot product of two binary vectors
    :param v1: list of term IDs
    :param v2: list of term IDs
    :return: Dot product
    """
    return len(set(v1).intersection(v2))


def squared_euclidean_dist(v1, v2):
    intersection = dot_product(v1, v2)
    u1 = len(v1) - intersection
    u2 = len(v2) - intersection
    return u1 + u2

if __name__ == "__main__":
    clf = SVM()
    clf.kernel_type = "linear"
    clf.load(sys.argv[2])
    sys_output = open(sys.argv[3], "w")
    with open(sys.argv[1], "r") as f:
        correct = 0
        total = 0
        for line in f:
            line = line.split()
            true_label = line[0]
            predicted_label, fx = clf.classify(parse_bvector(line[1:]))
            if str(predicted_label) == str(true_label):
                correct += 1
            total += 1
            sys_output.write(str(true_label) + " " + str(predicted_label) + " " + str(fx) + "\n")

        print("Accuracy: " + str(correct/float(total)))

