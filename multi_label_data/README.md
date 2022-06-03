# Multi-label datasets

The folder contains the preprocessed datasets in a JSON format, that were used in our experiments. For information about the original dataset source, preprocessing and JSON format check [multi_label_data_preprocessing](https://github.com/drndr/project_ds_textclass/tree/main/multi_label_data_preprocessing). Some files were zipped due to size limitations. Not included files are listed [here](#not-included-data-file-size--100-mb).

## Folder overview

For every multi-label dataset used in the paper there is a folder containing:
- train_data.json: containing the data used for training
- test_data.json: containing the data used for testing

Note: The NYT dataset has a additional val_data.json, which contains the data used for validation. Only for the NYT dataset we had access to a validation split provided by the [HiAGM paper](https://github.com/Alibaba-NLP/HiAGM).

## Not included data (due to file size > 100 MB or Lincense)
- RCV1-V2: test_data.json (1.08 GB)
- NYT: train_data.json , test_data.json (License required)
