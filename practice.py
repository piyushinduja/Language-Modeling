import os
from tqdm import tqdm
import pickle
import glob
import math as math

TRAIN_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\train'
TEST_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\test'
TRAIN1_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\train1'
TEST1_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\test1'

with open ('./data/vocab.pkl', 'rb') as f :
    vocab = pickle.load(f)

def convert_line2idx(line):
    line_data = []
    for charac in line:
        if charac not in vocab.keys():
            line_data.append(vocab["<unk>"])
        else:
            line_data.append(vocab[charac])
    return line_data


def convert_files2idx(path):  # Here is where I train the model i.e. making of nested dictionary "main_dict" as explained by you in class
    files = list(glob.glob(f"{path}/*.txt"))
    main_dict = {}
    for file in tqdm(files):
        data = []
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()
        
        for line in lines:
            toks = convert_line2idx(line)
            toks = [384, 384, 384] + toks + [384, 384, 384]
            data.append(toks)

        tokens = []
        for i in data:
            tokens.extend([[i[x:x+3], i[x+3]] for x in range(len(i)-3)])
        
        for i in range(len(tokens)):
            w = tokens[i][1]
            trigram = tokens[i][0]
            trigram = str(trigram)
            if trigram in main_dict.keys():
                if w in main_dict[trigram].keys():
                    main_dict[trigram][w] += 1
                else:
                    main_dict[trigram][w] = 1
            else:
                main_dict[trigram] = {w : 1}
    return main_dict

main_dict = convert_files2idx(TRAIN1_DATA)

def test(path, main_dict):  # Calculating perplexity on test data using "main_dict" from train data
    files = list(glob.glob(f"{path}/*.txt"))
    for file in files:
        data = []
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()
        
        for line in lines:
            toks = convert_line2idx(line)
            toks = [384, 384, 384] + toks + [384, 384, 384]
            data.append(toks)

        tokens = []
        for i in data:
            tokens.extend([[i[x:x+3], i[x+3]] for x in range(len(i)-3)])

        losses = []
        for i in range(len(tokens)):
            w = tokens[i][1]
            trigram = tokens[i][0]
            trigram = str(trigram)
            try:
                prob = (main_dict[trigram][w]+1)/(sum(main_dict[trigram].values())+len(vocab))
            except NameError:
                prob = (1)/(sum(main_dict[trigram].values())+len(vocab))
            except:
                prob = 1 / len(vocab)
            loss = -abs(math.log(prob, 2))
            losses.append(loss)
        avg_loss = sum(losses)/len(losses)
        perplexity = math.pow(2, avg_loss)
        print(perplexity)


test(TEST1_DATA, main_dict)
