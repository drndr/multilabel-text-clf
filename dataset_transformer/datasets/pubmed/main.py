import csv
import json
import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def generate_data_json():
    train_array = []
    test_array = []
    with open('pubmed.csv', mode='r', encoding="utf8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            if line_count != 0:
                if row["fold"] == "10":
                    train_array.append(
                        {"id": row["id"],
                         "text": ' '.join(row["title"].split()),
                         "labels": re.split(r'\t+', row["labels"]),
                         "fold": row["fold"]})
                if row["fold"] != "10":
                    test_array.append(
                        {"id": row["id"],
                         "text": ' '.join(row["title"].split()),
                         "labels": re.split(r'\t+', row["labels"]),
                         "fold": row["fold"]})
            line_count += 1

    print(line_count)
    print(len(train_array))
    print(len(test_array))
    print(len(test_array)+len(train_array))
    with open('test_data.json', 'w') as outfile:
        json.dump(test_array, outfile)
    with open('train_data.json', 'w') as outfile:
        json.dump(train_array, outfile)


def draw_label_distribution():
    labels = []
    folds = []

    with open('pubmed.csv', mode='r', encoding="utf8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            labels += re.split(r'\t+', row["labels"])
            folds.append(row["fold"])
            line_count += 1

    label_count = list(Counter(labels).values())
    label_count.sort(reverse=True)
    unique_labels = list(Counter(labels).keys())
    unique_labels.sort(reverse=True)

    x = np.array(unique_labels)
    y = np.array(label_count)

    plt.figure(figsize=(8, 10))

    barlist = plt.bar(np.arange(len(x)), y)
    barlist[0].set_color('#1f77b4')
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

    with open('pubmed.csv', mode='r', encoding="utf8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            labels = []
            if line_count == 0:
                line_count += 1
            labels += re.split(r'\t+', row["labels"])
            labels_dist.append(len(labels))
            line_count += 1

    print(Counter(labels_dist))
    unique_labels = Counter(labels_dist).keys()
    print(unique_labels)
    print(list(Counter(labels_dist).values()))
    label_count = [x / (line_count-1) for x in list(Counter(labels_dist).values())]
    f, ax = plt.subplots(figsize=(16, 10))
    plt.bar(unique_labels, label_count)
    plt.xticks(list(unique_labels), fontsize=12)
    plt.yticks(fontsize=18)
    plt.ylabel("Relative label frequency", fontsize=18)
    plt.title('n = ' + str(line_count-1), fontsize=18)
    plt.xlabel("# of labels", fontsize=18)
    plt.tight_layout()
    plt.savefig('label_per_doc_dist')


if __name__ == '__main__':
    generate_data_json()
    draw_label_distribution()
    draw_label_per_doc_dist()
