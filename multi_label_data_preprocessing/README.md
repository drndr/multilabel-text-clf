# Preprocessing for multi-label datasets

This code is used to generate a unified dataset format for the experiments. The sources for the original datasets and the used train-test split are referenced in [datasets](#datasets). The generated JSON format for most datasets can be found at [multi-label-data](https://github.com/drndr/project_ds_textclass/tree/main/multi_label_data) (Note: JSON for pubmed and RCV1-V2 is not included, because of file size). Additionally, scripts are included to generate plots for label distribution and label per document distribution for both multi-label and single-label datasets.

## Code overview

For every multi-label dataset used in the paper there is a folder containing:
- a main.py to generate the train-test split in JSON format (see [format](#json-format)), label distribution, label per document distribution
- the label distribution as png
- the label per document distribution as png

For every single-label dataset used in the paper there is a folder containing:
- a main.py to generate label distribution
- the label distribution as png


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
   - Source: Licensed
   - train-test split: From [HiAGM paper](https://github.com/Alibaba-NLP/HiAGM/tree/master/data)
- Amazon-531
   - Source: [Amazon used in TaxoClass](https://aclanthology.org/2021.naacl-main.335/)
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
