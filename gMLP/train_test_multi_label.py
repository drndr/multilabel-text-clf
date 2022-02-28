import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from transformers import BertTokenizer, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords

import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from torch import cuda
import logging

import matplotlib.pyplot as plt

from sklearn import metrics

##############################################################
# Experiment Setup
##############################################################
n_labels=298 # reuters 90, dbpedia 298
has_attention = True

TRAIN_BATCH_SIZE = 32  # 64 Econbiz, 32 rest
VALID_BATCH_SIZE = 32  # 64 Econbiz, 32 rest
LEARNING_RATE = 1e-4
num_epochs = 300

###############################################################
# Source Data
###############################################################
train_list = json.load(open("../multi_label_data/dbpedia/train_data.json",))
train_data = np.array(list(map(lambda x: (list(x.values())[:2]), train_list)),dtype=object)
train_labels= np.array(list(map(lambda x: list(x.values())[2], train_list)),dtype=object)
print(len(train_data))
test_list = json.load(open("../multi_label_data/dbpedia/test_data.json",))
test_data = np.array(list(map(lambda x: list(x.values())[:2], test_list)),dtype=object)
test_labels = np.array(list(map(lambda x: list(x.values())[2], test_list)),dtype=object)

################################################################
# Preprocess Labels
################################################################
label_encoder = MultiLabelBinarizer()
label_encoder.fit([*train_labels,*test_labels])
train_labels_enc = label_encoder.transform(train_labels)
test_labels_enc = label_encoder.transform(test_labels)

################################################################
# Create DataFrames
################################################################
train_df = pd.DataFrame()
train_df['text'] = train_data[:,1]
train_df['labels'] = train_labels_enc.tolist()

test_df = pd.DataFrame()
test_df['text'] = test_data[:,1]
test_df['labels'] = test_labels_enc.tolist()

print("Number of train texts ",len(train_df['text']))
print("Number of train labels ",len(train_df['labels']))
print("Number of test texts ",len(test_df['text']))
print("Number of test labels ",len(test_df['labels']))

###############################################################
# Remove Stopwords
###############################################################
#nltk.download('stopwords')
#stopwords_eng = stopwords.words('english')

#def preprocess_text(text):
#    pp_text = nltk.re.sub("[^a-zA-Z]", " ", text)  # remove remaining special characters
#    pp_text = pp_text.lower()
#    words = pp_text.split()
#    words = [word for word in words if not word in stopwords_eng]  # remove stop words
#    return words

###############################################################
# Define CustomDataset
###############################################################
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
        #text = preprocess_text(text)
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

###############################################################
# Train-Val-Test Split
###############################################################
MAX_LEN = 512
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
test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

###############################################################
# gMLP/aMLP Model
###############################################################
class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len, tiny_attention):
        super().__init__()
        self.has_tiny_attention = tiny_attention
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        if tiny_attention:
            self.act = nn.Identity()
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x,gate_res = None):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        if self.has_tiny_attention:
            v = v + gate_res
            out = u * self.act(v)
        else:
            out= u * v
        return out

class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len,tiny_attention,attn_dim=64):
        super().__init__()
        self.has_tiny_attention = tiny_attention
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn*2)

        #add attention
        if self.has_tiny_attention:
            self.attn = Attention(d_model, d_ffn, attn_dim)

        self.sgu = SpatialGatingUnit(d_ffn, seq_len, tiny_attention)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        if self.has_tiny_attention:
            gate_res=self.attn(x)
        x = F.gelu(self.channel_proj1(x))
        if self.has_tiny_attention:
            x = self.sgu(x,gate_res=gate_res)
        else:
            x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out

class gMLP(nn.Module):
    def __init__(self, d_model=512, d_ffn=1024, seq_len=512, num_layers=6, tiny_attention=False):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len,tiny_attention) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)

class gMLPForMultiLabelClassification(gMLP):
    def __init__(self, num_tokens=50000, d_model=512, d_ffn=1024, seq_len=512, num_layers=6, num_labels=5661, tiny_attention=False):
        super().__init__(d_model, d_ffn, seq_len, num_layers, tiny_attention)
        self.embed = nn.Embedding(num_tokens, d_model)
        self.to_vector = nn.Linear(d_model,num_labels)

    def forward(self, x):
        # Embedding layer
        embedding = self.embed(x)

        # gMLP layers
        out = self.model(embedding)

        # Transform for classifierhead
        final = out.mean(dim=1) #reference https://github.com/jaketae/g-mlp/blob/master/g_mlp/core.py
        # Classifier head
        head= self.to_vector(final)
        return head

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

############################################################
# Load Data
############################################################
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }      

training_loader = DataLoader(training_set,drop_last=True, **train_params)
validation_loader = DataLoader(validation_set, drop_last=True, **test_params)
test_loader = DataLoader(test_set, drop_last=True, **test_params)

############################################################
# Create Model
############################################################
device = 'cuda' if cuda.is_available() else 'cpu'
logging.basicConfig(level=logging.ERROR)

model = gMLPForMultiLabelClassification(num_layers=18,num_labels=n_labels,tiny_attention=has_attention)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=10,verbose=True)
model.to(device)

############################################################
# Plot Val loss
############################################################
def loss_plot(epochs, loss):
    plt.plot(epochs, loss, color='red', label='loss')
    plt.xlabel("epochs")
    plt.title("validation loss")
    plt.savefig("val_loss_dbpedia.png")

############################################################
# Train Model
############################################################
def train_model(start_epochs,  n_epochs,
                training_loader, validation_loader, model,optimizer):
  
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
        ids = data['ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)
        outputs = model(ids)
        #Backward
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

    ######################    
    # Validate the model #
    ######################
 
    model.eval()
    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))

      # calculate average losses
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      #scheduler.step(valid_loss)
      # print training/validation statistics 
      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
      loss_vals.append(valid_loss)
    # Plot loss
  loss_plot(np.linspace(1, n_epochs, n_epochs).astype(int), loss_vals)
  torch.save(model.state_dict(), "amlp_dbpedia_sw.pth")
  return model

trained_model = train_model(1, num_epochs, training_loader, validation_loader, model, optimizer)

####################################################################
# Test Model
####################################################################
def test(testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

outputs_nothresh, targets = test(test_loader)

targets=np.array(targets).astype(int)
outputs=np.where(np.array(outputs_nothresh) > 0.2, 1, 0)
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_avg = metrics.f1_score(targets, outputs, average='samples')
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print("Evaluation with 0.2 threshold")
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Samples) = {f1_score_avg}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")

outputs=np.where(np.array(outputs_nothresh) > 0.5, 1, 0)
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_avg = metrics.f1_score(targets, outputs, average='samples')
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print("Evaluation with 0.5 threshold")
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Samples) = {f1_score_avg}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")
