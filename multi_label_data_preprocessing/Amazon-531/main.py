import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def generate_test_data_json():
    labels = []
    with open("test/doc2labels.txt", encoding="utf8") as doc2labelsfile:
        doc2labels = doc2labelsfile.readlines()
        for label_line in doc2labels:
            doc2labels_list = label_line.split('\t')
            labels.append(doc2labels_list[1])

    label_names = []
    with open("test/labels.txt", encoding="utf8") as labelsfile:
        file = labelsfile.readlines()
        for label_line in file:
            labels_list = label_line.split('\t')
            label_names.append(labels_list[1])

    test_array = []
    with open("test/corpus.txt", encoding="utf8") as txtfile:
        lines = txtfile.readlines()
        line_count = 0
        for line in lines:
            doclabelsnames = []
            corpus_list = line.split('\t')
            doclabels = labels[int(corpus_list[0])]
            doclabelslist = doclabels.split(',')
            for doclabel in doclabelslist:
                doclabelsnames.append(label_names[int(doclabel)].rstrip())

            test_array.append(
                {"id": corpus_list[0],
                 "text": corpus_list[1].rstrip(),
                 "labels": doclabelsnames})
            line_count += 1

    print(len(test_array))

    with open('test/test_data.json', 'w') as outfile:
        json.dump(test_array, outfile)


def generate_train_data_json():
    labels = []
    with open("train/doc2labels.txt", encoding="utf8") as doc2labelsfile:
        doc2labels = doc2labelsfile.readlines()
        for label_line in doc2labels:
            doc2labels_list = label_line.split('\t')
            labels.append(doc2labels_list[1])

    label_names = []
    with open("train/labels.txt", encoding="utf8") as labelsfile:
        file = labelsfile.readlines()
        for label_line in file:
            labels_list = label_line.split('\t')
            label_names.append(labels_list[1])

    train_array = []
    with open("train/corpus.txt", encoding="utf8") as txtfile:
        lines = txtfile.readlines()
        line_count = 0
        for line in lines:
            doclabelsnames = []
            corpus_list = line.split('\t')
            doclabels = labels[int(corpus_list[0])]
            doclabelslist = doclabels.split(',')
            for doclabel in doclabelslist:
                doclabelsnames.append(label_names[int(doclabel)].rstrip())

            train_array.append(
                {"id": corpus_list[0],
                 "text": corpus_list[1].rstrip(),
                 "labels": doclabelsnames})
            line_count += 1

    print(len(train_array))

    with open('train/train_data.json', 'w') as outfile:
        json.dump(train_array, outfile)


def draw_label_distribution():
    labels = []
    with open("test/doc2labels.txt", encoding="utf8") as doc2labelsfile:
        doc2labels = doc2labelsfile.readlines()
        line_count = 0
        for label_line in doc2labels:
            doc2labels_list = label_line.split('\t')
            doclabels = doc2labels_list[1].split(',')
            for doclabel in doclabels:
                labels.append(doclabel.rstrip())
            line_count += 1
        print(line_count)

    with open("train/doc2labels.txt", encoding="utf8") as doc2labelsfile:
        doc2labels = doc2labelsfile.readlines()
        line_count = 0
        for label_line in doc2labels:
            doc2labels_list = label_line.split('\t')
            doclabels = doc2labels_list[1].split(',')
            for doclabel in doclabels:
                labels.append(doclabel.rstrip())
            line_count += 1
        print(line_count)

    unique_labels = list(Counter(labels).keys())
    label_count = list(Counter(labels).values())
    print(sum(Counter(labels).values()))
    label_count.sort(reverse=True)
    x = np.array(unique_labels)
    y = np.array(label_count)


    f, ax = plt.subplots(figsize=(8, 10))
    plt.bar(unique_labels, label_count)
    plt.yscale('log')
    plt.ylim(0, 15000000)
    plt.xticks([])
    plt.yticks(fontsize=18)
    plt.ylabel("# of occurrences in logarithmic scale", fontsize=18)
    plt.xlabel("Label", fontsize=18)
    plt.tight_layout()
    plt.savefig('label_dist')


def draw_label_per_doc_dist():
    labels_dist = []
    line_count = 0
    with open("test/doc2labels.txt", encoding="utf8") as doc2labelsfile:
        doc2labels = doc2labelsfile.readlines()
        for label_line in doc2labels:
            labels = []
            doc2labels_list = label_line.split('\t')
            doclabels = doc2labels_list[1].split(',')
            for doclabel in doclabels:
                labels.append(doclabel.rstrip())
            labels_dist.append(len(labels))
            line_count += 1
        print(line_count)

    with open("train/doc2labels.txt", encoding="utf8") as doc2labelsfile:
        doc2labels = doc2labelsfile.readlines()
        for label_line in doc2labels:
            labels = []
            doc2labels_list = label_line.split('\t')
            doclabels = doc2labels_list[1].split(',')
            for doclabel in doclabels:
                labels.append(doclabel.rstrip())
            labels_dist.append(len(labels))
            line_count += 1
        print(line_count)

    unique_labels = Counter(labels_dist).keys()
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
    generate_test_data_json()
    generate_train_data_json()
    draw_label_per_doc_dist()
    draw_label_distribution()

