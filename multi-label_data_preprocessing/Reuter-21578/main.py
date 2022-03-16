import json
from collections import Counter

import numpy as np
from nltk.corpus import reuters
import nltk
import matplotlib.pyplot as plt


def generate_train_data_json():
    nltk.download('reuters')
    documents = reuters.fileids()
    train_docs = list(filter(lambda doc: doc.startswith("train"),
                             documents))

    train_docs_array = []
    for train_doc in train_docs:
        train_docs_array.append(
            {"id": train_doc, "text": ' '.join(reuters.raw(train_doc).split()),
             "labels": reuters.categories(train_doc)})

    with open('train_data.json', 'w') as outfile:
        json.dump(train_docs_array, outfile)


def generate_test_data_json():
    nltk.download('reuters')
    documents = reuters.fileids()
    test_docs = list(filter(lambda doc: doc.startswith("test"),
                            documents))

    test_docs_array = []
    for test_doc in test_docs:
        test_docs_array.append(
            {"id": test_doc, "text": ' '.join(reuters.raw(test_doc).split()), "labels": reuters.categories(test_doc)})

    with open('test_data.json', 'w') as outfile:
        json.dump(test_docs_array, outfile)


def draw_label_distribution():
    nltk.download('reuters')
    documents = reuters.fileids()
    labels = []

    test_docs = list(filter(lambda doc: doc.startswith("test"),
                            documents))
    for test_doc in test_docs:
        for category in reuters.categories(test_doc):
            labels.append(category)

    train_docs = list(filter(lambda doc: doc.startswith("train"),
                             documents))
    for train_doc in train_docs:
        for category in reuters.categories(train_doc):
            labels.append(category)

    unique_labels = list(Counter(labels).keys())
    label_count = list(Counter(labels).values())
    label_count.sort(reverse=True)

    x = np.array(unique_labels)
    y = np.array(label_count)

    f, ax = plt.subplots(figsize=(8, 10))
    plt.bar(unique_labels, label_count)
    plt.yscale('log')
    plt.ylim(0, 15000000)
    plt.xticks([])
    plt.yticks(fontsize=18)
    plt.ylabel("# of occurences in logarithmic scale", fontsize=18)
    plt.xlabel("Label", fontsize=18)
    plt.tight_layout()
    plt.savefig('label_dist')


def draw_label_per_doc_dist():
    nltk.download('reuters')
    labels_dist = []
    line_count = 0
    documents = reuters.fileids()

    test_docs = list(filter(lambda doc: doc.startswith("test"),
                            documents))
    for test_doc in test_docs:
        labels = []
        for category in reuters.categories(test_doc):
            labels.append(category)
        labels_dist.append(len(labels))
        line_count += 1

    train_docs = list(filter(lambda doc: doc.startswith("train"),
                             documents))
    for train_doc in train_docs:
        labels = []
        for category in reuters.categories(train_doc):
            labels.append(category)
        labels_dist.append(len(labels))
        line_count += 1

    print(Counter(labels_dist))
    unique_labels = Counter(labels_dist).keys()
    print(unique_labels)
    print(list(Counter(labels_dist).values()))
    label_count = [x / line_count for x in list(Counter(labels_dist).values())]
    f, ax = plt.subplots(figsize=(8, 10))
    plt.bar(unique_labels, label_count)
    plt.xticks(list(unique_labels), fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Relative label frequency", fontsize=18)
    plt.title('n = ' + str(line_count), fontsize=18)
    plt.xlabel("# of labels", fontsize=18)
    plt.tight_layout()
    plt.savefig('label_per_doc_dist')


if __name__ == '__main__':
    generate_train_data_json()
    generate_test_data_json()
    draw_label_distribution()
    draw_label_per_doc_dist()
