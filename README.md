# Are We Really Making Much Progress? Bag-of-Words vs. Sequence vs. Graph vs. Hierarchy for Single- and Multi-Label Text Classification

This repository contains code to reproduce the results in our paper 'Are We Really Making Much Progress? Bag-of-Words vs. Sequence vs.
Graph vs. Hierarchy for Single- and Multi-Label Text Classification', which is an extension to the paper 'Bag-of-Words vs. Graph vs. Sequence in Text Classification: Questioning the Necessity of Text-Graphs and the Surprising Strength of a Wide MLP' published in ACL 2022.

The goal of this study was to compare a simple WideMLP model with current state-of-art graph-based and sequential models. We split the repository into subfolders based on the different models (see [folder structure](#folder-structure)).

### Folder structure:
    ├── HiAGM                                # Code for HiAGM
    ├── Transformers                         # Code for transformer models (BERT & DistilBERT)
    ├── WideMLP                              # Overview WideMLP
    │   ├── multilabel-processing            # Code for main WideMLP experiments reported in the paper
    │   ├── sparse-multilabel-processing     # Code for further WideMLP experiments on Pubmed dataset
    ├── gMLP                                 # Code for gMLP/aMLP
    ├── multi_label_data                     # Used multi-label datasets in JSON format
    ├── multi_label_data_preprocessing       # Code to preprocess datasets into JSON format     
    ├── single_label_data                    # Used single-label datasets
    └── README                               # Project structure overview

Every subfolder has a README with introductions on how to run the experiments.

The code for reproducing most single-label results can be found in following repositories:

[MLP and TextGCN](https://github.com/lgalke/text-clf-baselines)
[Tranformers](https://github.com/FKarl/text-classification)

### Contributors:
Andor Diera, Bao Xin Lin, Bhakti Khera, Tim Meuser, Tushar Singhal, Lukas Galke and Ansgar Scherp
