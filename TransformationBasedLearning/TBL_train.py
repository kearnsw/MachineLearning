from collections import Counter, defaultdict
import sys


class Corpus:
    def __init__(self, docs):
        self.docs = docs
        self.features = self.get_features()
        self.feat_map = dict()
        self.inv_feat_map = dict()
        self.label_map = dict()
        self.inv_label_map = dict()
        self.inverted_idx = defaultdict(list)

    def get_features(self):
        features = []
        for doc in self.docs:
            for feature in doc.word_counts.keys():
                features.append(feature)
        return list(set(features))

    def index(self):

        for idx, feature in enumerate(self.features):
            self.feat_map[feature] = idx
            self.inv_feat_map[idx] = feature

        label_idx = 0
        for idx, doc in enumerate(self.docs):
            doc.docID = idx  # Set document ID for inverted index
            if doc.label in self.label_map:
                doc.label_idx = self.label_map[doc.label]  # Set document label index
            else:
                self.label_map[doc.label] = label_idx
                self.inv_label_map[label_idx] = doc.label
                doc.label_idx = label_idx
                label_idx += 1

            for j, feat in enumerate(doc.feats):
                feat_idx = self.feat_map[feat]
                doc.feats[j] = feat_idx
                self.inverted_idx[feat_idx].append(doc.docID)


class Document:
    def __init__(self, text):
        self.docID = None
        self.raw_text = text
        self.label = None
        self.word_counts = Counter()
        self.feats = None
        self.label_idx = None
        self.predicted_label_idx = None

    def vectorize(self):
        tokens = self.raw_text.split()
        self.label = tokens[0]
        for word_count in tokens[1:]:
            word, count = word_count.split(":")
            self.word_counts[word] = count
        self.feats = list(self.word_counts.keys())
        return self


class TBL:
    def __init__(self):
        self.default = None
        self.transformations = []
        self.errors = None
        self.transformation_counter = None

    def train(self, corpus, min_gain):
        """
        Iterate over features and apply transformations until gain <= minimum gain
        :param corpus: corpus containing indexed documents
        :param min_gain: minimum gain before stopping transformation based learning
        :return: list of transformations
        """

        max_feat, from_label, to_label = None, None, None
        max_count = 0
        # Initial annotator
        for feat_idx in range(len(corpus.features)):
            self.transformation_counter = Counter()
            for doc_idx in corpus.inverted_idx[feat_idx]:
                doc = corpus.docs[doc_idx]
                if doc.label == doc.predicted_label_idx:
                    for label in corpus.label_map.keys():
                        self.transformation_counter[(doc.predicted_label_idx, label)] -= 1
                if doc.label != doc.predicted_label_idx:
                    self.transformation_counter[(doc.predicted_label_idx, doc.label)] += 1
            if self.transformation_counter:
                transform, count = self.transformation_counter.most_common(1)[0]
                if count > max_count:
                    from_label, to_label = transform
                    max_feat = feat_idx
                    max_count = count
        if max_feat is None:
            return 0

        print(max_count, corpus.inv_feat_map[max_feat], from_label, to_label)

        # Update predicted labels
        for doc_idx in corpus.inverted_idx[max_feat]:
            document = corpus.docs[doc_idx]
            if document.predicted_label_idx == from_label:
                document.predicted_label_idx = to_label

        self.transformations.append((max_feat, from_label, to_label))
        return self.get_errors([doc.predicted_label_idx for doc in corpus.docs], corpus.docs)

    @staticmethod
    def initial_annotator(docs, label_idx):
        for doc in docs:
            doc.predicted_label_idx = label_idx
        return [doc.predicted_label_idx for doc in docs]

    @staticmethod
    def get_errors(predicted_labels, docs):
        """
        Calculate the number of errors at end of iteration
        :param predicted_labels: list of label predictions for documents
        :param docs: list of indexed documents
        :return: number of errors, int
        """
        return sum([1 for idx, doc in enumerate(docs) if doc.label != predicted_labels[idx]])

    def load(self, tbl_model_file):
        f = open(tbl_model_file, "r")
        for line in f:
            line = line.split()
            if len(line) == 1:
                self.default = line[0]
            else:
                self.transformations.append(tuple(line))
        f.close()

    def classify(self, document, num_transformations):
        """
        Classifies a document using TBL transformation rules.

        :param document: the document to be classified
        :param num_transformations: the number of transformations to use in classifying the document selected from top
        :return: classification label, and list of transformations as tuple
        """
        # make sure that the num_transformations < how many are available
        if num_transformations > len(self.transformations):
            num_transformations = len(self.transformations)

        transformations = []
        document.predicted_label_idx = self.default
        for i in range(num_transformations):
            feat, from_label, to_label = self.transformations[i]
            if feat in document.feats and document.predicted_label_idx == from_label:
                document.predicted_label_idx = to_label
                transformations.append((feat, from_label, to_label))
        return document.predicted_label_idx, transformations

if __name__ == "__main__":
    train_data = open(sys.argv[1], "r")
    documents = [Document(doc).vectorize() for doc in train_data]
    train_data.close()

    cp = Corpus(documents)
    cp.index()

    tbl = TBL()
    predictions = tbl.initial_annotator(cp.docs, cp.docs[0].label)
    tbl.errors = tbl.get_errors(predictions, cp.docs)

    iterations = 0
    gain = tbl.errors
    while gain >= 1:
        errors = tbl.train(cp, sys.argv[3])
        gain = tbl.errors - errors
        tbl.errors = errors
        iterations += 1
    print(iterations)

    model_file = open(sys.argv[2], "w")
    model_file.write(cp.docs[0].label + "\n")
    for rule in tbl.transformations:
        max_feat, from_label, to_label = rule
        model_file.write(" ".join([cp.inv_feat_map[max_feat], from_label, to_label]) + "\n")
    model_file.close()
