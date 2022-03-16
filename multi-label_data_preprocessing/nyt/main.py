import xml.dom.minidom
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import re

"""
NYTimes Reference: https://catalog.ldc.upenn.edu/LDC2008T19
"""

sample_ratio = 0.02
train_ratio = 0.7
min_per_node = 200

month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
year_list = [1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]
# ROOT_DIR of LDC2008T19
ROOT_DIR = ''
label_f = 'nyt_label.vocab'
english_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
                     "won't", 'wouldn', "wouldn't"]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z(),!?\.\'\`]", " ", string)
    string = re.sub(r"/", " ", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\.", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", "  ", string)
    string = re.sub(r"!", "  ", string)
    string = re.sub(r"\(", "  ", string)
    string = re.sub(r"\)", "  ", string)
    string = re.sub(r"\?", "  ", string)
    string = re.sub(r"\s{2,}", " ", string)

    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# 2003-07
def read_nyt():
    labels = []
    labels_dist = []
    f = open('idnewnyt_train.json', 'r')
    ids = f.readlines()
    f.close()
    print(ids[:2])
    corpus_list = []
    f = open(label_f, 'r')
    label_vocab_s = f.readlines()
    f.close()
    label_vocab = []
    for label in label_vocab_s:
        label = label.strip()
        label_vocab.append(label)
    id_list = []
    for i in ids:
        id_list.append(int(i[13:-5]))
    print(id_list[:2])
    corpus1 = []

    for file_name in tqdm(ids):
        xml_path = ROOT_DIR + file_name.strip()
        try:
            sample = {}
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement
            tags = root.getElementsByTagName('p')
            text = ''
            for tag in tags[1:]:
                text += tag.firstChild.data
            if text == '':
                continue
            text = clean_str(text)
            text = [word.lower() for word in text.split() if word not in english_stopwords]
            path = file_name.split('/', 1)
            id = path[1].split('.')
            sample['id'] = id[0]
            sample['text'] = ' '.join(text)
            sample_label = []
            tags = root.getElementsByTagName('classifier')
            for tag in tags:
                type = tag.getAttribute('type')
                if type != 'taxonomic_classifier':
                    continue
                hier_path = tag.firstChild.data
                hier_list = hier_path.split('/')
                if len(hier_list) < 3:
                    continue
                for l in range(1, len(hier_list) + 1):
                    label = '/'.join(hier_list[:l])
                    if label == 'Top':
                        continue
                    if label not in sample_label and label in label_vocab:
                        sample_label.append(label)
                        labels.append(label)
            sample['labels'] = sample_label
            labels_dist.append(len(sample_label))
            corpus1.append(sample)
        except:
            print(xml_path)
            print('Something went wrong...')
            continue
    print('idnewnyt_train.json', len(corpus1))
    f = open('doc' + 'idnewnyt_train.json', 'w')
    f.write('[')
    for line in corpus1:
        doc = json.dumps(line)
        f.write(doc)
        if line != corpus1[-1]:
            f.write(',')
    f.write(']')
    f.close()

    f = open('idnewnyt_val.json', 'r')
    ids = f.readlines()
    f.close()
    print(ids[:2])
    corpus_list = []
    f = open(label_f, 'r')
    label_vocab_s = f.readlines()
    f.close()
    label_vocab = []
    for label in label_vocab_s:
        label = label.strip()
        label_vocab.append(label)
    id_list = []
    for i in ids:
        id_list.append(int(i[13:-5]))
    print(id_list[:2])
    corpus3 = []

    for file_name in tqdm(ids):
        xml_path = ROOT_DIR + file_name.strip()
        try:
            sample = {}
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement
            tags = root.getElementsByTagName('p')
            text = ''
            for tag in tags[1:]:
                text += tag.firstChild.data
            if text == '':
                continue
            text = clean_str(text)
            text = [word.lower() for word in text.split() if word not in english_stopwords]
            path = file_name.split('/', 1)
            id = path[1].split('.')
            sample['id'] = id[0]
            sample['text'] = ' '.join(text)
            sample_label = []
            tags = root.getElementsByTagName('classifier')
            for tag in tags:
                type = tag.getAttribute('type')
                if type != 'taxonomic_classifier':
                    continue
                hier_path = tag.firstChild.data
                hier_list = hier_path.split('/')
                if len(hier_list) < 3:
                    continue
                for l in range(1, len(hier_list) + 1):
                    label = '/'.join(hier_list[:l])
                    if label == 'Top':
                        continue
                    if label not in sample_label and label in label_vocab:
                        sample_label.append(label)
                        labels.append(label)
            sample['labels'] = sample_label
            labels_dist.append(len(sample_label))
            corpus3.append(sample)
        except:
            print(xml_path)
            print('Something went wrong...')
            continue
    print('idnewnyt_val.json', len(corpus3))
    f = open('doc' + 'idnewnyt_val.json', 'w')
    f.write('[')
    for line in corpus3:
        doc = json.dumps(line)
        f.write(doc)
        if line != corpus3[-1]:
            f.write(',')
    f.write(']')
    f.close()

    f = open('idnewnyt_test.json', 'r')
    ids = f.readlines()
    f.close()
    print(ids[:2])
    corpus_list = []
    f = open(label_f, 'r')
    label_vocab_s = f.readlines()
    f.close()
    label_vocab = []
    for label in label_vocab_s:
        label = label.strip()
        label_vocab.append(label)
    id_list = []
    for i in ids:
        id_list.append(int(i[13:-5]))
    print(id_list[:2])
    corpus2 = []

    for file_name in tqdm(ids):
        xml_path = ROOT_DIR + file_name.strip()
        try:
            sample = {}
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement
            tags = root.getElementsByTagName('p')
            text = ''
            for tag in tags[1:]:
                text += tag.firstChild.data
            if text == '':
                continue
            text = clean_str(text)
            text = [word.lower() for word in text.split() if word not in english_stopwords]
            path = file_name.split('/', 1)
            id = path[1].split('.')
            sample['id'] = id[0]
            sample['text'] = ' '.join(text)
            sample_label = []
            tags = root.getElementsByTagName('classifier')
            for tag in tags:
                type = tag.getAttribute('type')
                if type != 'taxonomic_classifier':
                    continue
                hier_path = tag.firstChild.data
                hier_list = hier_path.split('/')
                if len(hier_list) < 3:
                    continue
                for l in range(1, len(hier_list) + 1):
                    label = '/'.join(hier_list[:l])
                    if label == 'Top':
                        continue
                    if label not in sample_label and label in label_vocab:
                        sample_label.append(label)
                        labels.append(label)
            sample['labels'] = sample_label
            labels_dist.append(len(sample_label))
            corpus2.append(sample)
        except:
            print(xml_path)
            print('Something went wrong...')
            continue
    print('idnewnyt_test.json', len(corpus2))
    f = open('doc' + 'idnewnyt_test.json', 'w')
    f.write('[')
    for line in corpus2:
        doc = json.dumps(line)
        f.write(doc)
        if line != corpus2[-1]:
            f.write(',')
    f.write(']')
    f.close()

    line_count = len(corpus1) + len(corpus2) + len(corpus3)
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

    print(Counter(labels_dist))
    unique_labels = Counter(labels_dist).keys()
    print(unique_labels)
    print(list(Counter(labels_dist).values()))
    label_count = [x / (line_count) for x in list(Counter(labels_dist).values())]
    f, ax = plt.subplots(figsize=(16, 10))
    plt.bar(unique_labels, label_count)
    plt.xticks(list(unique_labels), fontsize=12)
    plt.yticks(fontsize=18)
    plt.ylabel("Relative label frequency", fontsize=18)
    plt.title('n = ' + str(line_count), fontsize=18)
    plt.xlabel("# of labels", fontsize=18)
    plt.tight_layout()
    plt.savefig('label_per_doc_dist')


if __name__ == '__main__':
    read_nyt()
