"""
File: multilabel_data.py
Description: Stuff to facilitate sparse multilabel classification
Author: Lukas Galke
Email: git@lpag.de
Date: 2022-01-31
"""
import torch
import numpy as np
import scipy.sparse as sp

import tokenizers


def multilabel_collate_for_mlp(examples: list):
    # iterate through list of examples
    offset = 0
    flat_docs, offsets = [], []
    label_rows = []
    for doc, labels in examples:
        if isinstance(doc, tokenizers.Encoding):
            doc = doc.ids
        offsets.append(offset)
        flat_docs.extend(doc)
        offset += len(doc)

        label_rows.append(labels)

    # convert to tensor
    x = torch.tensor(flat_docs)
    offsets = torch.tensor(offsets)

    y = torch.tensor(sp.vstack(label_rows).toarray(), dtype=torch.float)

    return x, offsets, y


class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, documents: [[int]], label_indicator_matrix):
        self.x = np.asarray(documents, dtype=object)
        self.y = label_indicator_matrix  # may be sparse

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
