# Importing stock ml libraries
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DebertaTokenizer, DebertaModel, DebertaPreTrainedModel
import logging
logging.basicConfig(level=logging.ERROR)
import numpy as np
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]="3"

# Source Data
dataset = "Econbiz"   #[ 'R21578', 'RCV1-V2', 'Econbiz', 'Amazon-531', 'DBPedia-298','NYT AC','GoEmotions']
labels = 5661                #[90,103,5661,512,298,166,28]
epochs = 15                #[15,15,15,15,5,15,5]
train_list = json.load(open("../datasets/econbiz/train_data.json")) #change the dataset folder name [ 'reuters', 'rcv1-v2', 'econbiz', 'amazon', 'dbpedia','nyt','goemotions']
train_data = np.array(list(map(lambda x: (list(x.values())[:2]), train_list)),dtype=object)
train_labels= np.array(list(map(lambda x: list(x.values())[2], train_list)),dtype=object)
test_list = json.load(open("../datasets/econbiz/test_data.json")) #change dataset folder name
test_data = np.array(list(map(lambda x: list(x.values())[:2], test_list)),dtype=object)
test_labels = np.array(list(map(lambda x: list(x.values())[2], test_list)),dtype=object)


# Preprocess Labels
from sklearn.preprocessing import MultiLabelBinarizer
label_encoder = MultiLabelBinarizer()
label_encoder.fit([*train_labels,*test_labels])
train_labels_enc = label_encoder.transform(train_labels)
test_labels_enc = label_encoder.transform(test_labels)

# Create DataFrames
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
train_df.head()
test_df


#  Setting up the device for GPU usage
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DebertaConfig



# Sections of config
# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = epochs
LEARNING_RATE = 5e-05
config = DebertaConfig()
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

# Define CustomDataset
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
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
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


# Creating the dataset and dataloader for the neural network 
# Train-Val-Test Split
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

# Load Data
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0
               }
training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **test_params)
testing_loader = DataLoader(testing_set, **test_params)

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class DebertaClass(DebertaPreTrainedModel):
    def __init__(self):
        super(DebertaClass, self).__init__(config)
        self.deberta = transformers.DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base",num_labels=labels,problem_type="multi_label_classification")
        #self.pooler = transformers.ContextPooler(config)
        #output_dim = self.pooler.output_dim
        #self.classifier = nn.Linear(output_dim, num_labels)
        #drop_out = getattr(config, "cls_dropout", None)
        #drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        #self.dropout = transformers.StableDropout(drop_out)

        # Initialize weights and apply final processing
        #self.post_init()

    def forward(self, ids, mask,token_type_ids):
        #outputs = self.deberta(ids, attention_mask=mask,token_type_ids=token_type_ids)
        #encoder_layer = outputs[0]
        #pooled_output = self.pooler(encoder_layer)
        #pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)
        logits = self.deberta(ids, attention_mask=mask,token_type_ids=token_type_ids).logits
        return logits

model = DebertaClass()
model.to(device)

# Define Loss function 
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Plot Val loss
import matplotlib.pyplot as plt
def loss_plot(epochs, loss):
    plt.plot(epochs, loss, color='red', label='loss')
    plt.xlabel("epochs")
    plt.title("validation loss")
    plt.savefig(dataset + "_val_loss.png")
   
# Train Model
def train_model(start_epochs, n_epochs,
                training_loader, validation_loader, model, optimizer):
    loss_vals = []
    for epoch in range(start_epochs, n_epochs + 1):
        train_loss = 0
        valid_loss = 0

        ######################
        # Train the model #
        ######################

        model.train()
        print('############# Epoch {}: Training Start   #############'.format(epoch))
        for batch_idx, data in enumerate(training_loader):
            optimizer.zero_grad()

            # Forward
            ids = data['ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask,token_type_ids)
            # Backward
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
                ids = data['ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
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
        # Plot loss
    loss_plot(np.linspace(1, n_epochs, n_epochs).astype(int), loss_vals)
    return model


trained_model = train_model(1, epochs, training_loader, validation_loader, model, optimizer)


def validation(testing_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask,token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

# Test Model
outputs, targets = validation(testing_loader)
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_avg = metrics.f1_score(targets, outputs, average='samples')
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Samples) = {f1_score_avg}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")

#Save results
import sys
with open(dataset + "_results.txt", "w") as f:
    print(f"F1 Score (Samples) = {f1_score_avg}",f"Accuracy Score = {accuracy}",f"F1 Score (Micro) = {f1_score_micro}",f"F1 Score (Macro) = {f1_score_macro}", file=f)
