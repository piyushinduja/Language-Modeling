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
TRAIN1_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\train1'
DEV_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\dev'
DEV1_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\dev1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

VOCAB_SIZE = len(vocab)
LEARNING_RATE = 0.0001
NUM_LAYERS = 2


class LSTM(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, 50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=200, num_layers=layers, batch_first=True)
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
    x_data, y_data, train_dict = convert_files2idx(TRAIN1_DATA, vocab)
    x_data = torch.Tensor(x_data).to(device).to(torch.int64)
    y_data = torch.Tensor(y_data).to(device).to(torch.int64)
    train_dataset = TensorDataset(x_data, y_data)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Dev dataset and data loader
    x_dev, y_dev, dev_dict = convert_files2idx(DEV1_DATA, vocab)
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
    model = LSTM(layers=NUM_LAYERS)
    loss_func = nn.CrossEntropyLoss(weight=weights.to(device).to(torch.float32), ignore_index=384, reduce=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(5):
        print("Epoch: ", epoch+1)

        train_losses = []
        train_perplexities = []
        for x, y in tqdm(train_dataloader):
            print('y', y.shape)
            model.train()
            pred = model(x)
            pred = pred.reshape(pred.shape[0], pred.shape[2], pred.shape[1])
            # print('pred', pred.shape)
            loss = loss_func(pred, y)
            # print('loss', loss.shape)
#############################################################
            mask = (y!=384).float()


            non_pad_loss = (loss * mask)
            non_pad_loss_sum = non_pad_loss.sum()

            individual_losses = [loss for loss in non_pad_loss]
            mean_individual_losses = [loss.mean() for loss in individual_losses]
            p = [(math.e)**mean_loss for mean_loss in mean_individual_losses]
            train_perplexities.extend(p)

            no_of_non_pad_tokens = mask.sum()

            mean_loss = non_pad_loss_sum / no_of_non_pad_tokens

            loss = mean_loss
            
            train_losses.append(loss.cpu().item())
##############################################################

            # y_copy = torch.clone(y)
            # y_copy[y_copy==384] = 0.0
            # y_copy[y_copy!=0.0] = 1.0
            # non_pad_count = torch.sum(y_copy, dim=1)
            # non_pad_loss_sum = torch.sum(loss, dim=0)
            # mean_loss = non_pad_count / non_pad_loss_sum
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_losses.append(loss.cpu().item())
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
                #############################################################
                mask = (y!=384).float()


                non_pad_loss = (loss * mask)
                non_pad_loss_sum = non_pad_loss.sum()

                individual_losses = [loss for loss in non_pad_loss]
                mean_individual_losses = [loss.mean() for loss in individual_losses]
                p = [(math.e)**mean_loss for mean_loss in mean_individual_losses]
                dev_perplexities.extend(p)

                no_of_non_pad_tokens = mask.sum()

                mean_loss = non_pad_loss_sum / no_of_non_pad_tokens

                loss = mean_loss
                
                dev_losses.append(loss.cpu().item())
    ##############################################################
                # dev_losses.append(loss.item())
        print('Dev Loss: ', np.average(dev_losses))
        print('Dev Perplexity: ', (sum(dev_perplexities)/len(dev_perplexities)))


if __name__ == "__main__":
    main()


# def create_dataset(path):
#     os.chdir(path)
#     main_list = []
#     for file in tqdm(os.listdir()):  # Remove tqdm
#         file_path = f'{TRAIN_DATA}/{file}'
#         f = open(file_path, 'r', encoding='utf8')
#         f1 = str(f.read().encode('utf8'))
#         f1 = f1[2:len(f1)-1]
#         f1 = list(f1) # Strings and paragraphs to char
#         pad_len = 500 - f1.__len__() % 500
#         for i in range(pad_len): # Appending [PAD] chars
#             f1.append('[PAD]')
#         for i in range(len(f1)): # Replacing low freq chars with <unk>
#             if f1[i] not in list(vocab.keys()):
#                 f1[i] = '<unk>'
#             # f1[i] = 
#         f2 = [f1[x:x+500] for x in range(0, len(f1), 500)]
#         main_list.extend(f2)
#     print(main_list[0])


# pad_len = 500 - file_data.__len__() % 500
# for i in range(pad_len):
#     file_data.append(384)
# f2 = [file_data[x:x+500] for x in range(0, len(file_data), 500)]
# data.extend(f2)
