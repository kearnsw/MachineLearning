import sys
from maxent_classify import MaxEnt
from calc_emp_exp import Corpus

def calc_model_exp(corpus, classifier):
    for doc in corpus.docs:
        if classifier:
            topic_probs = classifier.classify(doc)
        else:
            num_topics = len(corpus.topics)
            topic_probs = [(topic, 1/num_topics) for topic in corpus.topics]
        for word in doc.word_counts:
            for topic, prob in topic_probs:
                corpus.expectation[topic][word] += prob / float(corpus.num_docs)
                corpus.raw_counts[topic][word] += 1

if __name__ == "__main__":
    training_data = sys.argv[1]
    output_file = sys.argv[2]

    cp = Corpus()
    cp.load(training_data)

    if len(sys.argv) >= 4:
        model_file = sys.argv[3]
        clf = MaxEnt()
        clf.load(model_file)
        calc_model_exp(cp, clf)
    else:
        calc_model_exp(cp, classifier=None)

    cp.print_model_expectation(sys.argv[2])
