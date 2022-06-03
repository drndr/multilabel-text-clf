import csv
import json
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def generate_test_data_json():
    emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                "desire", "disappointment", "disapproval", "disgust",
                "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
                "optimism", "pride", "realization", "relief",
                "remorse", "sadness", "surprise", "neutral"]
    train_array = []
    with open("test.tsv", encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        line_count = 0
        for line in tsvreader:
            labels = []
            label_string = line[1]
            label_list = label_string.split(',')
            for label in label_list:
                labels.append(emotions[int(label)])
            train_array.append(
                {"id": line[2],
                 "text": line[0],
                 "labels": labels})
            line_count += 1

    with open('test_data.json', 'w') as outfile:
        json.dump(train_array, outfile)


def generate_train_data_json():
    emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                "desire", "disappointment", "disapproval", "disgust",
                "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
                "optimism", "pride", "realization", "relief",
                "remorse", "sadness", "surprise", "neutral"]
    train_array = []
    with open("train.tsv", encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        line_count = 0
        for line in tsvreader:
            labels = []
            label_string = line[1]
            label_list = label_string.split(',')
            for label in label_list:
                labels.append(emotions[int(label)])
            train_array.append(
                {"id": line[2],
                 "text": line[0],
                 "labels": labels})
            line_count += 1

    with open('train_data.json', 'w') as outfile:
        json.dump(train_array, outfile)


def draw_label_distribution():
    emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                "desire", "disappointment", "disapproval", "disgust",
                "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
                "optimism", "pride", "realization", "relief",
                "remorse", "sadness", "surprise", "neutral"]
    labels = []

    with open("train.tsv", encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        line_count = 0
        for line in tsvreader:
            label_string = line[1]
            label_list = label_string.split(',')
            for label in label_list:
                labels.append(emotions[int(label)])
            line_count += 1

    with open("test.tsv", encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        line_count = 0
        for line in tsvreader:
            label_string = line[1]
            label_list = label_string.split(',')
            for label in label_list:
                labels.append(emotions[int(label)])
            line_count += 1

    unique_labels = list(Counter(labels).keys())
    print(len(unique_labels))
    label_count = list(Counter(labels).values())
    label_count.sort(reverse=True)
    print(sum(Counter(labels).values()))
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
    emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                "desire", "disappointment", "disapproval", "disgust",
                "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
                "optimism", "pride", "realization", "relief",
                "remorse", "sadness", "surprise", "neutral"]
    labels_dist = []
    line_count = 0

    with open("train.tsv", encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            labels = []
            label_string = line[1]
            label_list = label_string.split(',')
            for label in label_list:
                labels.append(emotions[int(label)])
            labels_dist.append(len(labels))
            line_count += 1
    print("train: " + str(line_count))

    with open("test.tsv", encoding="utf8") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            labels = []
            label_string = line[1]
            label_list = label_string.split(',')
            for label in label_list:
                labels.append(emotions[int(label)])
            labels_dist.append(len(labels))
            line_count += 1
    print("test: " + str(line_count))

    print(Counter(labels_dist))
    unique_labels = Counter(labels_dist).keys()
    print(unique_labels)
    print(list(Counter(labels_dist).values()))
    label_count = [x / line_count for x in list(Counter(labels_dist).values())]
    f, ax = plt.subplots(figsize=(8, 10))
    plt.bar(unique_labels, label_count)
    plt.xticks(list(unique_labels), fontsize=12)
    plt.yticks(fontsize=18)
    plt.ylabel("Relative label frequency", fontsize=18)
    plt.title('n = ' + str(line_count), fontsize=18)
    plt.xlabel("# of labels", fontsize=18)
    plt.tight_layout()
    plt.savefig('label_per_doc_dist')


if __name__ == '__main__':
    generate_test_data_json()
    generate_train_data_json()
    draw_label_distribution()
    draw_label_per_doc_dist()
