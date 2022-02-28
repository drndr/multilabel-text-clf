import json
import os.path as osp
#import nltk
from collections import Counter
#nltk.download('stopwords')
#from nltk.corpus import stopwords

import numpy as np
import torch
from joblib import Memory
from tqdm import tqdm


CACHE_DIR = 'tmp/cache'
MEMORY = Memory(CACHE_DIR, verbose=2)
VALID_DATASETS = ['reuters', 'dbpedia', 'goemotions', 'econbiz', 'pubmed', 'amazon', 'rcv1-v2', 'nyt']


@MEMORY.cache
def load_word_vectors(path, unk_token=None):
    vocab = dict()
    vectors = []
    with open(path, mode='r') as myfile:
        for i, line in tqdm(enumerate(myfile)):
            word, *vector_str = line.strip().split(' ')
            if len(vector_str) == 1:
                print(f"[load_word_vectors] Ignoring row {i + 1}: {line}")
                continue

            # Parse word vector
            vector = torch.tensor([float(val) for val in vector_str])

            vocab[word] = len(vocab)
            vectors.append(vector)

    embedding = torch.stack(vectors)

    return vocab, embedding


@MEMORY.cache(ignore=['n_jobs'])
def load_data(key, tokenizer, dataset_folder, max_length=None, construct_textgraph=False, n_jobs=1,
              force_lowercase=False):
    assert key in VALID_DATASETS, f"{key} not in {VALID_DATASETS}"
    print("Loading raw documents")

    train_list = json.load(open(osp.join(dataset_folder, key, 'train_data' + '.json')))
    train_data = np.array(list(map(lambda x: (list(x.values())[1]), train_list)), dtype=object)
    N_train = len(train_data)

    test_list = json.load(open(osp.join(dataset_folder, key, 'test_data' + '.json')))
    test_data = np.array(list(map(lambda x: (list(x.values())[1]), test_list)), dtype=object)
    N_test = len(test_data)

    print(f"Number of Train docs: {N_train}")
    print(f"Number of Test  docs: {N_test}")

    raw_documents = np.append(test_data, train_data)
    #stopwords_eng = stopwords.words('english')
    #preprocessed_docs = [nltk.re.sub("[^a-zA-Z]", " ", raw_doc) for raw_doc in raw_documents]
    #preprocessed_docs = [preprocessed_doc.lower() for preprocessed_doc in preprocessed_docs]
    #docs_without_stopwords = []
    #for preprocessed_doc in preprocessed_docs:
    #    doc_without_stopwords = [word for word in preprocessed_doc.split() if not word in stopwords_eng]
    #    docs_without_stopwords.append(doc_without_stopwords)
    #N = len(preprocessed_docs)

    N = len(raw_documents)

    #print("First few raw_documents", *docs_without_stopwords[:5], sep='\n')

    train_mask, test_mask = torch.zeros(N, dtype=torch.bool), torch.zeros(N, dtype=torch.bool)
    print("Loading document metadata...")
    test_labels = np.array(list(map(lambda x: (list(x.values())[2]), test_list)), dtype=object)
    train_labels = np.array(list(map(lambda x: (list(x.values())[2]), train_list)), dtype=object)
    labels = np.concatenate((test_labels, train_labels), axis=0)
    # labels = np.concatenate(np.append(test_labels.data, train_labels.data), axis=0)


    # unique_labels = list(Counter(labels).keys())

    # raw_documents, labels, train_mask, test_mask defined

    if max_length:
        print(f"Encoding documents with max_length={max_length}...")
        # docs = [tokenizer.encode(doc_without_stopwords, max_length=max_length) for doc_without_stopwords in docs_without_stopwords]
        # docs = tokenizer(raw_documents, truncation=True, max_length=max_length)
        docs = [tokenizer.encode(raw_doc, max_length=max_length) for raw_doc in raw_documents]
    else:
        print(f"Encoding documents without max_length")
        # docs = [tokenizer.encode(doc_without_stopwords) for doc_without_stopwords in docs_without_stopwords]
        docs = [tokenizer.encode(raw_doc) for raw_doc in raw_documents]

    ## LABELS
    print("Encoding labels...")
    from sklearn.preprocessing import MultiLabelBinarizer
    print("Labels shape:", labels.shape)
    mlb = MultiLabelBinarizer(sparse_output=True).fit(labels)
    print(f"Found {mlb.classes_} classes.")
    label_ids = mlb.transform(labels)
    print("Label ids shape:", label_ids.shape)

    ## MASKS, IMPORTANT: first TEST then TRAIN
    train_mask = torch.cat([torch.zeros(N_test, dtype=torch.bool), torch.ones(N_train, dtype=torch.bool)],
                           dim=0)
    test_mask = ~train_mask
    print("Train check:", train_mask.sum())
    print("Test check:", test_mask.sum())

    # label2index = {label: idx for idx, label in enumerate(set(labels))}

    # label_ids = []
    # idx = 0
    # for array in test_labels:
    #     label_names = np.array(array)
    #     array_ids = np.empty(len(unique_labels))
    #     array_ids.fill(0)
    #     test_mask[idx] = True
    #     idx += 1
    #     for label_name in label_names:
    #         for label, index in label2index.items():
    #             if label_name == label:
    #                 array_ids[index] = 1
    #     label_ids.append(array_ids)

    # for array in train_labels:
    #     label_names = np.array(array)
    #     array_ids = np.empty(len(unique_labels))
    #     array_ids.fill(0)
    #     train_mask[idx] = True
    #     idx += 1
    #     for label_name in label_names:
    #         for label, index in label2index.items():
    #             if label_name == label:
    #                 array_ids[index] = 1
    #     label_ids.append(array_ids)

    # return docs, label_ids, train_mask, test_mask, label2index
    return docs, label_ids, train_mask, test_mask, mlb.classes_
