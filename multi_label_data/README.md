# Multi-label datasets

The folder contains the preprocessed datasets in a JSON format, that were used in our experiments. For information about the original dataset source, preprocessing and JSON format check [multi_label_data_preprocessing](https://github.com/drndr/project_ds_textclass/tree/main/multi_label_data_preprocessing). Not included data, because of exceeding file size are listed [here](#not-included-data-file-size--100-mb).

## Folder overview

For every multi-label dataset used in the paper there is a folder containing:
- train_data.json: containing the data used for training
- test_data.json: containing the data used for testing

## Not included data (file size > 100 MB)
- DBPedia: train_data.json (162 MB)
- EconBiz: train_data.json (171 MB)
- RCV1-V2: test_data.json (1.08 GB)
- Pubmed: train_data.json (4.43 GB), test_data.json (259 MB)
