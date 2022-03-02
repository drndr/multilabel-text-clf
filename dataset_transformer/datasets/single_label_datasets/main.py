from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def draw_label_distribution(path: str, name: str):
    labels = []

    with open(path) as file:
        lines = 0
        for line in file:
            text = line.split()
            labels.append(text[2])
            lines += 1

    print(lines)
    unique_labels = list(Counter(labels).keys())
    print(len(unique_labels))
    label_count = list(Counter(labels).values())
    label_count.sort(reverse=True)
    print(label_count[0])
    x = np.array(unique_labels)
    y = np.array(label_count)

    f, ax = plt.subplots(figsize=(8, 10))
    plt.bar(unique_labels, label_count)
    plt.yscale('log')
    plt.axis([None, None, 1, 10000])
    plt.xticks([])
    plt.yticks(fontsize=18)
    plt.ylabel("# of occurences in logarithmic scale", fontsize=18)
    plt.xlabel("Label", fontsize=18)
    plt.tight_layout()
    plt.savefig(name)


if __name__ == '__main__':
    draw_label_distribution('20ng/20ng.txt', '20ng')
    draw_label_distribution('mr/mr.txt', 'mr')
    draw_label_distribution('ohsumed/ohsumed.txt', 'ohsumed')
    draw_label_distribution('r8/R8.txt', 'r8')
    draw_label_distribution('r52/R52.txt', 'r52')
