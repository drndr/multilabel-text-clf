# HiAGM: Hierarchy-Aware Global Model for Hierarchical Text Classification
This repository implements the hierarchy-aware structure encoders for mutual interaction between label space and text features. 
The dataset splits of NYTimes (New York Times), Reuters V2, Amazon and DBpedia are proposed in this repository.


### Hierarchy-Aware Global Model
The hierarchy-aware global model improves the conventional text classification model with prior knowledge of the predefined hierarchical structure.
The project folder consists of following parts:
+ config: config files (json format)
+ data: data dir, could be changed in config file (with sample data)
+ data_modules: Dataset / DataLoader / Collator / Vocab
+ helper: Configure / Hierarchy_Statistic / Logger / Utils
+ models: StructureModel / EmbeddingLayer / TextEncoder / TextPropagation (HiAGM-TP) / Multi-Label Attention (HiAGM-LA)
+ train_modules: Criterions / EvaluationMetrics / Trainer
+ preprocess_labels.py: create taxonomy file from label hierarchy and label ids
+ remove_duplicate.py: remove duplicate from taxonomy file
+ HiAGM_format.ipynb: transform to json format file {'token': List[str], 'label': List[str]}

#### Hierarchy-Aware Structure Encoder
+ Bidirectional TreeLSTM: weighted_tree_lstm.py & tree.py
+ Hierarchy-GCN: graphcnn.py

### Setup
+ Python >= 3.6
+ torch >= 0.4.1
+ numpy >= 1.17.4

### Preprocess
#### data_modules.preprocess
+ clean stopwords

#### Prior Probability
+ helper.hierarchical_statistic
+ Note that first change the Root.child List 
+ calculate the prior probability between parent-child pair in train dataset


### Train
```bash
python train.py config/gcn-rcv1-v2.json
```
+ optimizer -> train.set_optimizer: default torch.optim.Adam
+ learning rate decay schedule callback -> train_modules.trainer.update_lr
+ earlystop callback -> train.py 
+ Hyper-parameters are set in config.train

## Citation
Please cite ACL 2020 paper if using HiAGM:

    @article{jie2020hierarchy,  
     title={Hierarchy-Aware Global Model for Hierarchical Text Classification},  
     author={Jie Zhou, Chunping Ma, Dingkun Long, Guangwei Xu, Ning Ding, Haoyu Zhang, Pengjun Xie, Gongshen Liu},  
     booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
     year={2020}  
    }

