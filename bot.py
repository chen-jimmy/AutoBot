import os
import discord
from discord.ext import commands
from discord.ext.commands import has_permissions, MissingPermissions
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

import torch
import pickle
from torch import nn
import numpy as np
import pandas as pd
import os
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

def input_text(input):
    ids = "0002bcb3da6cb337"
    text = input
    labels = [0,0,0,0,0,0]
    examples = []
    examples.append(InputExample(ids, text, labels=labels))
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

print("download bert")
device = torch.device(type='cuda')
pretrained_weights = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
basemodel = BertModel.from_pretrained(pretrained_weights)
basemodel.to(device)
print("bert downloaded")
seq_len = 256
embed_num = seq_len 
embed_dim = basemodel.config.hidden_size 
dropout = basemodel.config.hidden_dropout_prob
kernel_num = 3
kernel_sizes = [2,3,4]
num_labels = 6

def load_model():
	from torch import load
	from os import path
	r = KimCNN(embed_num, embed_dim, dropout=dropout, kernel_num=kernel_num, kernel_sizes=kernel_sizes, num_labels=num_labels)
	r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'KimCNN.th')))
	return r

model = load_model()
model.eval()
model = model.to(device)

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print('evaluating...')

def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s

def evaluate(input):
    y_pred = []
    examples = input_text(input)
    features = get_features_from_examples(examples, seq_len, tokenizer)
    input_dataset = get_dataset_from_features(features)
    input_sampler = SequentialSampler(input_dataset)
    input_dataloader = DataLoader(input_dataset, sampler=input_sampler, batch_size=1)

    for step, batch in enumerate(input_dataloader):
        batch = tuple(t.to(device) for t in batch)
        val_input_ids, val_input_mask, val_segment_ids, val_label_ids = batch
        with torch.no_grad():
            val_inputs,_ = basemodel(val_input_ids, val_segment_ids, val_input_mask)
            logits = model(val_inputs)
        y_pred.append(logits)
    y_pred = torch.cat(y_pred, dim=0).float().cpu().detach().numpy()
    text = tokenizer.decode(input_dataset[0][0], skip_special_tokens=True)
    preds = dict(zip(labels, sigmoid(y_pred[0])))
    for label in preds:
        print(label, ': ', preds[label])
    return preds

evaluate("test")

#client = discord.Client()
bot = commands.Bot(command_prefix='$')

infractions = {}
strictness = .1

def is_bad(message):
	preds = evaluate(message)
	print("threat: " + str(preds["threat"]))
	return preds["threat"] > strictness

@bot.event
async def on_message(message):
	if message.author.bot:
		return

	if message.content == 'ping':
		await message.channel.send('pong')
		return

	if is_bad(message.content):
		if message.author.name in infractions:
			infractions[message.author.name] = infractions[message.author.name] + 1
		else:
			infractions[message.author.name] = 1

		await message.channel.send('This is your {}th warning {}'.format(infractions[message.author.name], message.author.name))
		return

	await bot.process_commands(message)

@bot.command()
async def test(ctx, arg):
	await ctx.send(arg)

@bot.command(pass_context=True)
@has_permissions(administrator=True)
async def set_strictness(ctx, arg):
	global strictness
	try:
		value = float(arg)
		strictness = value	
		await ctx.send("Set strictness to " + str(strictness))
	except:
		await ctx.send("Not a valid float!")

@set_strictness.error
async def set_strictness_error(ctx, error):
	await ctx.send("oops")

bot.run(TOKEN)
