# Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP

## Project Submission for Project Data Science on Very Large Datasets 2021/22

This repository contains code to reproduce the results in our paper 'Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP'. The goal of this project was to compare a simple WideMLP model with current state-of-art models (HiAGM, BERT, DistilBERT, gMLP) on multi-label classification and compare the results to the single-label classification case, where WideMLP showed a strong performance. We split the repository into subfolders based on the different models (see [folder structure](#folder-structure)). Every subfolder has a README with introductions on how to run experiments.

### Folder structure:
    ├── HiAGM                                # Code for HiAGM
    ├── Transformers                         # Code for transformer models (BERT & DistilBERT)
    ├── WideMLP                              # Overview WideMLP
    │   ├── multilabel-processing            # Code for main WideMLP experiments reported in the paper
    │   ├── sparse-multilabel-processing     # Code for further WideMLP experiments on Pubmed dataset
    ├── gMLP                                 # Code for gMLP
    ├── multi-label_data_preprocessing       # Code to preprocess datasets into JSON format     
    ├── multi_label_data                     # Used multi-label datasets in JSON format
    ├── single_label_data                     # Used single-label datasets
    └── README                               # Project structure overview
