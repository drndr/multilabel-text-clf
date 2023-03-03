import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, einsum

from torch import cuda

import logging
from tqdm import tqdm
import os.path as osp
import os

import matplotlib.pyplot as plt

from sklearn import metrics

from timeit import default_timer as timer

os.environ["CUDA_VISIBLE_DEVICES"]="1"

###############################################################################
# Experiment Setup
###############################################################################
dataset = '20ng'    #[ '20ng', 'R8', 'R52', 'ohsumed', 'mr']
n_labels = 20 #[20,8,52,23,2]
num_epochs = 10

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
LEARNING_RATE = 1e-5

###############################################################################
# Load Data
###############################################################################
print("Loading raw documents")
with open(osp.join('../single_label_data', 'corpus', dataset+'.txt'), 'rb') as f:
    raw_documents = [line.strip().decode('latin1') for line in tqdm(f)]

#print("First few raw_documents", *raw_documents[:5], sep='\n')
N = len(raw_documents)

train_labels = []
test_labels = []
train_data = []
test_data = []

print("Loading document metadata...")
doc_meta_path = osp.join('../single_label_data', dataset+'.txt')
with open(doc_meta_path, 'r') as f:
    i=0
    for idx, line in tqdm(enumerate(f)):
        __name, train_or_test, label = line.strip().split('\t')
        if 'test' in train_or_test:
            test_labels.append(label)
            test_data.append(raw_documents[i])
        elif 'train' in train_or_test:
            train_labels.append(label)
            train_data.append(raw_documents[i])
        else:
            raise ValueError("Doc is neither train nor test:"
                             + doc_meta_path + ' in line: ' + str(idx+1))
        i+=1
print("Encoding labels...")
label2index = {label: idx for idx, label in enumerate(set([*train_labels, *test_labels]))}
train_label_ids = [label2index[train_label] for train_label in tqdm(train_labels)]
test_label_ids = [label2index[test_label] for test_label in tqdm(test_labels)]

train_labels = train_label_ids
train_data = train_data
test_labels = test_label_ids
test_data = test_data

################################################################################
# Create DataFrames
################################################################################
train_df = pd.DataFrame()
train_df['text'] = train_data
train_df['labels'] = train_labels

test_df = pd.DataFrame()
test_df['text'] = test_data
test_df['labels'] = test_labels

print("Number of train texts ",len(train_df['text']))
print("Number of train labels ",len(train_df['labels']))
print("Number of test texts ",len(test_df['text']))
print("Number of test labels ",len(test_df['labels']))

################################################################################
# Create Custom Dataset
################################################################################
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.targets = dataframe.labels
        self.max_len = max_len

    def __len__(self):
      return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

#############################################################################
# Create Train-Test Split
#############################################################################
train_size = 0.8
train_dataset = train_df.sample(frac=train_size,random_state=200)
valid_dataset = train_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
test_dataset  = test_df.reset_index(drop=True)

print("TRAIN Dataset: {}".format(train_dataset.shape))
print("VAL Dataset: {}".format(valid_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(valid_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

#############################################################################
# BERT Large Model
#############################################################################

class BERTLargeClass(torch.nn.Module):
    def __init__(self):
        super(BERTLargeClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-large-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(1024, n_labels)

    def forward(self, ids, mask,token_type_ids):
        outputs = self.l1(ids, attention_mask=mask,token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.l2(pooled_output)
        output = self.l3(pooled_output)
        return output
   
criterion = nn.CrossEntropyLoss()
###############################################################################
# Create DataLoader and Model
###############################################################################

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
              
training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)
testing_loader = DataLoader(testing_set, **test_params)

device = 'cuda' if cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.ERROR)

model = BERTLargeClass()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
model.to(device)

###############################################################################
# Train Model
###############################################################################
def train_model(start_epochs,  n_epochs,
                training_loader, model,optimizer):
  
  loss_vals = []
  for epoch in range(start_epochs, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    ######################    
    # Train the model #
    ######################

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(training_loader):
        optimizer.zero_grad()
        #Forward
        ids = data['ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        outputs = model(ids, mask,token_type_ids)
        #Backward
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

    ######################    
    # Validate the model #
    ######################

    model.eval()
    with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)

                loss = criterion(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))

            # calculate average losses
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)
            # print training/validation statistics
            print('Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
              ))
            loss_vals.append(valid_loss)
  return model

start = timer() # Start measuring time for Train and Inference
trained_model = train_model(1, num_epochs, training_loader, model, optimizer)

################################################################################
# Test Model
################################################################################
def test(testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask,token_type_ids)
            preds = outputs.argmax(axis=1)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(preds.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

outputs, targets = test(testing_loader)
end = timer() # Stop measuring time for Train and Inference

targets=np.array(targets).astype(int)
outputs=np.array(outputs).astype(int)
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print("Evaluation")
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")

print("Train+Inference time in seconds: ",end - start)
print("Train+Inference time in minutes: ",(end-start)/60)