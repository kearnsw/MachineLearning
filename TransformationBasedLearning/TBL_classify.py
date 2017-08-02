from TBL_train import Document, TBL
import sys


if __name__ == "__main__":

    # load test data
    test_data = open(sys.argv[1], "r")
    documents = [Document(doc).vectorize() for doc in test_data]
    test_data.close()

    # load model
    tbl = TBL()
    tbl.load(sys.argv[2])

    # classify test data
    output_file = open(sys.argv[3], "w")
    num_transformations = int(sys.argv[4])
    predicted = []
    for idx, doc in enumerate(documents):
        label, transformations = tbl.classify(doc, num_transformations)
        output_file.write("array" + str(idx) + ": " + doc.label + " " + label + " ")
        for transformation in transformations:
            output_file.write(" ".join(transformation) + " ")
        output_file.write("\n")
        predicted.append(label)
    print("Accuracy: " + str((1 - (tbl.get_errors(predicted, documents)/len(documents)))*100) + "%")
    output_file.close()

