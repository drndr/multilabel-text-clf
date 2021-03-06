## Transformer Models - BERT and DistilBERT 

This folder contains are implementation of pre-trained BERT and DistilBERT models from Transformers (Hugging Face) customized multi-label text classification.

## Get up and running

1. Make sure that the datasets are placed into a subfolder `./data/multi_label_data`.

2. Check for dependencies `numpy`, `torch`, `transformers`, `pandas`,`sklearn`,`tqdm` and `matplotlib`


## Code overview

- In `bert_model_multi_label.py`and `distilbert_model_multi_label.py` you can find the implementation for the BERT and DistilBERT models for multi label classification.
  The experimental setup can be configured in the beginning of the script:
    - valid datasets are: {'amazon', 'dbpedia', 'econbiz', 'nyt', 'reuters', 'rcv1-v2', 'goemotions'}
    - the corresponding label number can be found in the python files.

## Running experiments

The scripts `bert_model_multi_label.py` and `distilbert_model_multi_label.py` are the main entry point for running an experiment. Variables at the beginning of the scripts should be set accordingly to datasets and label number.

## References

    @inproceedings{wolf-etal-2020-transformers,
    title = {Transformers: State-of-the-Art Natural Language Processing},
    author = {Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
    month = {oct},
    year = {2020},
    address = {Online},
    publisher = {Association for Computational Linguistics},
    url = {https://www.aclweb.org/anthology/2020.emnlp-demos.6},
    pages = {38--45}
    }

