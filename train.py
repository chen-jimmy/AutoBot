import torch
import pickle
from torch import nn
import numpy as np
import pandas as pd
from transformers import *
from sklearn.metrics import roc_curve, auc
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, random_split

class InputExample(object):
    def __init__(self, id, text, labels=None):
        self.id = id
        self.text = text
        self.labels = labels

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def get_train_examples(train_file):
    train_df = pd.read_csv(train_file)
    ids = train_df['id'].values
    text = train_df['comment_text'].values
    labels = train_df[train_df.columns[2:]].values
    examples = []
    for i in range(len(train_df)):
        examples.append(InputExample(ids[i], text[i], labels=labels[i]))
    return examples

def get_features_from_examples(examples, max_seq_len, tokenizer):
    features = []
    for i,example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:(max_seq_len - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(tokens)
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        label_ids = [float(label) for label in example.labels]
        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_ids=label_ids))
    return features

def get_dataset_from_features(features):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.float)
    dataset = TensorDataset(input_ids,
                            input_mask,
                            segment_ids,
                            label_ids)
    return dataset

class KimCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, dropout=0.1, kernel_num=3, kernel_sizes=[2,3,4], num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (k, self.embed_dim)) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(len(self.kernel_sizes)*self.kernel_num, self.num_labels)
        
    def forward(self, inputs, labels=None):
        output = inputs.unsqueeze(1)
        output = [nn.functional.relu(conv(output)).squeeze(3) for conv in self.convs]
        output = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        output = self.dropout(output)
        logits = self.classifier(output)
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
print("downloading bert")
device = torch.device(type='cuda')
pretrained_weights = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
basemodel = BertModel.from_pretrained(pretrained_weights)
basemodel.to(device)
print("downloaded bert")
seq_len = 1024
train_file = 'train.csv'
train_examples = get_train_examples(train_file)
train_features = get_features_from_examples(train_examples, seq_len, tokenizer)
train_dataset = get_dataset_from_features(train_features)


train_val_split = 0.1
train_size = int(len(train_dataset)*(1-train_val_split))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

batch = 256
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch)
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch)
print("dataloader done")
embed_num = seq_len 
embed_dim = basemodel.config.hidden_size 
dropout = basemodel.config.hidden_dropout_prob
kernel_num = 3
kernel_sizes = [2,3,4]
num_labels = 6

model = KimCNN(embed_num, embed_dim, dropout=dropout, kernel_num=kernel_num, kernel_sizes=kernel_sizes, num_labels=num_labels)
model.to(device)

def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'KimCNN.th'))

lr = 3e-5
epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

max_accuracy = 0.0
for i in range(epochs):
    print('-----------EPOCH #{}-----------'.format(i+1))
    print('training...')
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        with torch.no_grad():
            inputs,_ = basemodel(input_ids, segment_ids, input_mask)
        loss = model(inputs, label_ids)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
    
    y_true = []
    y_pred = []

    model.eval()
    print('evaluating...')
    save_model(model)
    for step, batch in enumerate(val_dataloader):
        batch = tuple(t.to(device) for t in batch)
        val_input_ids, val_input_mask, val_segment_ids, val_label_ids = batch
        with torch.no_grad():
            val_inputs,_ = basemodel(val_input_ids, val_segment_ids, val_input_mask)
            logits = model(val_inputs)
        y_true.append(val_label_ids)
        y_pred.append(logits)

    y_true = torch.cat(y_true, dim=0).float().cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).float().cpu().detach().numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i,label in enumerate(labels):
        fpr[label], tpr[label], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])
    
    average_accuracy = 0.0
    print('ROC AUC per label:')
    for label in labels:
        average_accuracy += roc_auc[label]
        print(label, ': ', roc_auc[label])
    average_accuracy = average_accuracy/6
    print("Average: ", average_accuracy)

    if average_accuracy >= max_accuracy:
        max_accuracy = average_accuracy
        save_model(model)