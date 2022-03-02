import csv
import json
from collections import Counter

import numpy as np
import re
import matplotlib.pyplot as plt
from ast import literal_eval


def generate_data_json():
    train_array = []
    test_array = []
    clean = re.compile('<.*?>')
    with open("rcv1_v2.csv", encoding="utf8") as csvFile:
        csvReader = csv.DictReader(csvFile)
        for rows in csvReader:
            text = re.search("<text>[\\s\\S]*?</text>", rows['text'])
            text1 = text[0].replace('"', '')
            text2 = text1.replace('\n', '')
            text3 = text2.replace('\t', '')
            text4 = text3.replace('\\', '')
            text5 = re.sub(clean, '', str(text4))
            if 26150 < int(rows["id"]) >= 2286:
                train_array.append(
                    {"id": rows["id"],
                     "text": text5,
                     "labels": literal_eval(rows['topics'])})
            if 26151 > int(rows["id"]) <= 810596:
                test_array.append(
                    {"id": rows["id"],
                     "text": text5,
                     "labels": literal_eval(rows['topics'])})

    print(len(test_array))
    print(len(train_array))

    with open('train_data.json', 'w') as outfile:
        json.dump(test_array, outfile)

    with open('test_data.json', 'w') as outfile:
        json.dump(train_array, outfile)


def draw_label_distribution():
    labels = []
    with open("rcv1_v2.csv", encoding="utf8") as csvFile:
        csvReader = csv.DictReader(csvFile)
        line_count = 0
        for rows in csvReader:
            labels.append(literal_eval(rows['topics']))
            line_count += 1
        print(line_count)

    labels = list(np.concatenate(labels).flat)
    unique_labels = list(Counter(labels).keys())
    print(len(unique_labels))
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
    plt.ylabel("# of occurrences in logarithmic scale", fontsize=18)
    plt.xlabel("Label", fontsize=18)
    plt.tight_layout()
    plt.savefig('label_dist')


def draw_label_per_doc_dist():
    labels_dist = []
    line_count = 0
    with open("rcv1_v2.csv", encoding="utf8") as csvFile:
        csvReader = csv.DictReader(csvFile)
        line_count = 0
        for rows in csvReader:
            labels_dist.append(len(literal_eval(rows['topics'])))
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
    generate_data_json()
    draw_label_distribution()
    draw_label_per_doc_dist()
