# WideMLP

This folder contains the code to run experiments on WideMLP. The code is based on [text-clf-baselines](https://github.com/lgalke/text-clf-baselines). 

## Folder overview

- [multilabel-processing](https://github.com/drndr/project_ds_textclass/tree/main/WideMLP/multilabel-processing): This folder is the main entry point for reproducing our results and running further experiments.
- [sparse-multilabel-processing](https://github.com/drndr/project_ds_textclass/tree/main/WideMLP/sparse-multilabel-processing): This folder was used to experiment with the Pubmed dataset. Pubmed contains over 12 million documents and 27,773 classes. Because of Pubmeds size, we ran into memory-problems for generating the document-label matrix. To solve this problem, we implemented the same setup as in [multilabel-processing](https://github.com/drndr/project_ds_textclass/tree/main/WideMLP/multilabel-processing), but generated a sparse matrix instead of a dense matrix. Other models ran into similar problems, thats why we dropped this dataset from the paper. The code for experimenting with Pubmed is just included for completeness and further experiments. 
