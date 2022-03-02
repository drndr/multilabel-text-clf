# Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP -- Code for Dataset Formatter

## Code overview

For every multi-label dataset used in the paper there is a folder containing:
- the original dataset (see [Datasets](#datasets))(on Serverc)
- the train-test split (train/test-data.json) as JSON format (see [JSON format](#JSON-format))(on Serverc)
- the label distribution as png
- the label per document distribution as png
- a main.py to generate train-test split, label distribution, label per document distribution
- a main.py to generate label distribution for single label datasets

NOTE: Datasets and JSON is not included, because of file size.

## Datasets
- Econbiz
   - Source: [Econbiz on Kaggle](https://www.kaggle.com/hsrobo/multi-label-classification-evaluation-template/data?select=econbiz.csv)
   - train-test split: Folds 0-9 for testing, Fold 10 for training
- Pubmed
   - Source: [Pubmed on Kaggle](https://www.kaggle.com/hsrobo/multi-label-classification-evaluation-template/data?select=pubmed.csv)
   - train-test split: Folds 0-9 for testing, Fold 10 for training
- NYT AC
   - Source: [Link to NYT AC (Licensed)](https://catalog.ldc.upenn.edu/LDC2008T19)
   - train-test split: From [HiAGM paper](https://github.com/Alibaba-NLP/HiAGM/tree/master/data)
- RCV1-V2
   - Source: Licensed (Provided by Lukas)
   - train-test split: From [HiAGM paper](https://github.com/Alibaba-NLP/HiAGM/tree/master/data)
- Amazon-531
   - Source: [Amazon used in TacoClass](https://aclanthology.org/2021.naacl-main.335/)
   - train-test split: split from TaxoClass (asked authors)
- DBPedia-298
   - Source: [DBPedia used in TaxoClass](https://aclanthology.org/2021.naacl-main.335/)
   - train-test split: split from TaxoClass (asked authors)
- Reuters-21578
   - Source: [Reuters from NLTK](https://www.nltk.org/book/ch02.html)
   - train-test split: split provided by NLTK
  

## JSON format

The JSON format contains an array of documents. For each document an ID, the document text and the labels are saved.
- Example:
```json
[
    {
        "id":"doc1",
        "text":"This is an example text",
        "labels":[
            "label1",
            "label2"
        ]
    }
]
```