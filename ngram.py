import pickle
import glob
import math as math
from tqdm import tqdm

TRAIN_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\train'
TRAIN1_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\train1'
TEST1_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\test1'
TEST_DATA = r'C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\data\test'

with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


def convert_line2idx(line, vocab):
    line_data = []
    for charac in line:
        if charac not in vocab.keys():
            line_data.append(vocab["<unk>"])
        else:
            line_data.append(vocab[charac])
    return line_data


def convert_files2idx(path, vocab):
    num_para = 0
    files = list(glob.glob(f"{path}/*.txt"))
    main_dict = {}
    for file in tqdm(files):
        data = []
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()

        for line in lines:
            toks = convert_line2idx(line, vocab)
            toks = [384, 384, 384] + toks
            data.append(toks)

        tokens = []
        for i in data:
            tokens.extend([[i[x:x + 3], i[x + 3]] for x in range(0, len(i) - 3)])
        # num_para = num_para + len(tokens)

        for i in range(len(tokens)):
            w = tokens[i][1]
            trigram = tokens[i][0]
            trigram = str(trigram)
            if trigram in main_dict.keys():
                if w in main_dict[trigram].keys():
                    main_dict[trigram][w] += 1
                else:
                    main_dict[trigram][w] = 1
                    num_para += 1
            else:
                main_dict[trigram] = {w: 1}
                num_para += 1
    return main_dict, num_para


main_dict, num_para = convert_files2idx(TRAIN_DATA, vocab)


def test(path, vocab, main_dict):
    total_lines = 0
    total_perp = 0
    files = list(glob.glob(f"{path}/*.txt"))
    for file in tqdm(files):
        with open(file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            total_lines += len(lines)

        for line in lines:
            toks = convert_line2idx(line, vocab)
            toks = [384, 384, 384] + toks
            tokens = [[toks[x:x + 3], toks[x + 3]] for x in range(0, len(toks) - 3)]
            losses = []
            for i in range(len(tokens)):
                w = tokens[i][1]
                trigram = tokens[i][0]
                trigram = str(trigram)
                if trigram in main_dict.keys():
                    if w in main_dict[trigram].keys():
                        prob = (main_dict[trigram][w] + 1) / (sum(main_dict[trigram].values()) + len(vocab))
                    else:
                        prob = 1 / (sum(main_dict[trigram].values()) + len(vocab))
                else:
                    prob = 1 / len(vocab)

                loss = -math.log(prob, math.e)
                losses.append(loss)
            avg_loss = sum(losses) / len(line)
            perplexity = math.e ** avg_loss
            total_perp += perplexity
    return total_lines, total_perp


lines, perp = test(TEST_DATA, vocab, main_dict)
print(perp/lines)
print(num_para)
