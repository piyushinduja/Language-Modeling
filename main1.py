import os
from tqdm import tqdm
import pickle
import glob
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import numpy as np
import math

TRAIN_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\train'
TEST_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\test'
DEV_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\dev'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

VOCAB_SIZE = len(vocab)
LEARNING_RATE = 0.0001
NUM_LAYERS = 2
EPOCHS = 5


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, 50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=200, num_layers=NUM_LAYERS, batch_first=True)
        self.lin1 = nn.Linear(200, 275)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(275, VOCAB_SIZE)

    def forward(self, x):
        embeds = self.embeddings(x)
        out, (hidd, cont) = self.lstm(embeds)
        out1 = torch.reshape(out, (out.shape[0]*out.shape[1], out.shape[2])) # (32000, 200)
        l1 = self.lin1(out1)
        relu = self.relu(l1)
        l2 = self.lin2(relu)
        l2 = torch.reshape(l2, (out.shape[0], out.shape[1], l2.shape[1]))
        return l2


def convert_line2idx(line, vocab):
    line_data = []
    for charac in line:
        if charac not in vocab.keys():
            line_data.append(vocab["<unk>"])
        else:
            line_data.append(vocab[charac])
    return line_data


def convert_files2idx(path, vocab):
    x_data = []
    y_data = []
    dict1 = {i:0 for i in range(len(vocab))}
    files = list(glob.glob(f"{path}/*.txt"))
    for file in files:
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()

        toks = []
        for line in lines:
            tok = convert_line2idx(line, vocab)
            toks.extend(tok)
        dict1.update(Counter(toks))
        pad_len = 500 - len(toks) % 500
        toks = toks + [384] * pad_len
        x_toks = [toks[x:x + 500] for x in range(0, len(toks), 500)]
        x_data.extend(x_toks)
        toks = toks + [384]
        y_toks = [toks[x:x + 500] for x in range(1, len(toks), 500)]
        y_data.extend(y_toks)
    return x_data, y_data, dict(sorted(dict1.items()))


def main():
    # Train dataset and data loader
    print('creating train data')
    x_data, y_data, train_dict = convert_files2idx(TRAIN_DATA, vocab)
    x_data = torch.Tensor(x_data).to(device).to(torch.int64)
    y_data = torch.Tensor(y_data).to(device).to(torch.int64)
    train_dataset = TensorDataset(x_data, y_data)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Dev dataset and data loader
    print('creating dev data')
    x_dev, y_dev, dev_dict = convert_files2idx(DEV_DATA, vocab)
    x_dev = torch.Tensor(x_dev).to(device).to(torch.int64)
    y_dev = torch.Tensor(y_dev).to(device).to(torch.int64)
    dev_dataset = TensorDataset(x_dev, y_dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=True)

    # Loss weights
    c_count = list(train_dict.values())
    v_count = np.sum(c_count)
    weights = 1 - (c_count/v_count)
    weights = torch.tensor(weights)

    # Model, optimizer and loss function
    model = LSTM().to(device)
    loss_func = nn.CrossEntropyLoss(weight=weights.to(device).to(torch.float32), ignore_index=384, reduce=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_model = {'perp':10000, 'epoch':-1, 'parameters':0, 'model':{}, 'optimizer':{}}
    for epoch in range(EPOCHS):
        print("Epoch: ", epoch+1)

        train_losses = []
        train_perplexities = []
        for x, y in tqdm(train_dataloader):
            model.train()
            pred = model(x)
            pred = pred.reshape(pred.shape[0], pred.shape[2], pred.shape[1])
            loss = loss_func(pred, y)
            non_pad_indices = (y!=384).float()
            loss = loss*non_pad_indices
            loss_sum = torch.sum(loss, dim=1)
            rows_mean_loss = []
            for row_losses in non_pad_loss:
                rows_mean_loss.append(torch.mean(row_losses, dim=0))
            for mean_losses in rows_mean_loss:
                row_perp = (math.e)**mean_losses
                train_perplexities.append(row_perp)
            non_pad_tokens_count = torch.sum(non_pad_indices, dim=1)
            mean_loss = loss_sum / non_pad_tokens_count
            mean_loss = torch.mean(mean_loss, dim=0)
            loss = mean_loss
            train_losses.append(mean_loss.cpu().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Training Loss: ', np.average(train_losses))
        print('Training Perplexity: ', (sum(train_perplexities)/len(train_perplexities)))

        dev_losses = []
        dev_perplexities = []
        for x, y in tqdm(dev_dataloader):
            model.eval()
            with torch.no_grad():
                pred = model(x)
                pred = pred.reshape(pred.shape[0], pred.shape[2], pred.shape[1])
                loss = loss_func(pred, y)
                non_pad_indices = (y!=384).float()
                non_pad_loss = loss*non_pad_indices
                non_pad_loss_sum = torch.sum(non_pad_loss, dim=1)
                rows_mean_loss = []
                for row_losses in non_pad_loss:
                    rows_mean_loss.append(torch.mean(row_losses, dim=0))
                for mean_losses in rows_mean_loss:
                    row_perp = (math.e)**mean_losses
                    dev_perplexities.append(row_perp)
                non_pad_tokens_count = torch.sum(non_pad_indices, dim=1)
                mean_loss = non_pad_loss_sum / non_pad_tokens_count
                mean_loss = torch.mean(mean_loss, dim=0)
                loss = mean_loss
                dev_losses.append(mean_loss.cpu().item())
        print('Dev Loss: ', np.average(dev_losses))
        print('Dev Perplexity: ', (sum(dev_perplexities)/len(dev_perplexities)))
        dev_perp = sum(dev_perplexities)/len(dev_perplexities)
        if dev_perp < best_model['perp']:
            best_model['perp'] = dev_perp
            best_model['epoch'] = epoch+1
            best_model['parameters'] = sum(p.numel() for p in model.parameters())
            best_model['model'] = model.state_dict()
            best_model['optimizer'] = optimizer.state_dict()

    
    torch.save({
        'model_state_dict':best_model['model'],
        'optim_state_dict':best_model['optimizer'],
        'epoch':best_model['epoch'],
        'perp':best_model['perp'],
        'parameters':best_model['parameters']
    }, f'best_model_piyush')

    return train_dict


# if __name__ == "__main__":
#     train_dict = main()

checkpoint = torch.load(r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\best_model', map_location=torch.device('cpu'))
print(checkpoint['parameters'])


# # Calculating Test Perplexity
# # Test dataset and data loader
# print('creating test data')
# x_test, y_test, test_dict = convert_files2idx(TEST_DATA, vocab)
# x_test = torch.Tensor(x_test).to(device).to(torch.int64)
# y_test = torch.Tensor(y_test).to(device).to(torch.int64)
# test_dataset = TensorDataset(x_test, y_test)
# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# # Loss weights
# c_count = list(train_dict.values())
# v_count = np.sum(c_count)
# weights = 1 - (c_count/v_count)
# weights = torch.tensor(weights)

# checkpoint = torch.load(r"C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\best_model", map_location=torch.device('cpu'))
# model = LSTM().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# loss_func = nn.CrossEntropyLoss(weight=weights.to(device).to(torch.float32), ignore_index=384, reduce=False)


# test_losses = []
# test_perplexities = []
# for x, y in tqdm(test_dataloader):
#             model.eval()
#             with torch.no_grad():
#                 pred = model(x)
#                 pred = pred.reshape(pred.shape[0], pred.shape[2], pred.shape[1])
#                 loss = loss_func(pred, y)
#                 non_pad_indices = (y!=384).float()
#                 non_pad_loss = loss*non_pad_indices
#                 non_pad_loss_sum = torch.sum(non_pad_loss, dim=1)
#                 rows_mean_loss = []
#                 for row_losses in non_pad_loss:
#                     rows_mean_loss.append(torch.mean(row_losses, dim=0))
#                 for mean_losses in rows_mean_loss:
#                     row_perp = (math.e)**mean_losses
#                     test_perplexities.append(row_perp)
#                 non_pad_tokens_count = torch.sum(non_pad_indices, dim=1)
#                 mean_loss = non_pad_loss_sum / non_pad_tokens_count
#                 mean_loss = torch.mean(mean_loss, dim=0)
#                 loss = mean_loss
#                 test_losses.append(mean_loss.cpu().item())

# print('Test Loss: ', np.average(test_losses))
# print('Test Perplexity: ', (sum(test_perplexities)/len(test_perplexities)))
