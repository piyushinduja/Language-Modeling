import torch
import torch.nn as nn
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
k = list(vocab.keys())
i_to_word = {i:k[i] for i in range(len(k))}

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

    def forward(self, x, hidd, cont):
        embeds = self.embeddings(x)
        embeds = torch.unsqueeze(embeds, dim=0)
        out, (hidd, cont) = self.lstm(embeds, (hidd, cont))
        out1 = torch.reshape(out, (out.shape[0]*out.shape[1], out.shape[2]))
        l1 = self.lin1(out1)
        relu = self.relu(l1)
        l2 = self.lin2(relu)
        l2 = torch.reshape(l2, (out.shape[0], out.shape[1], l2.shape[1]))
        return l2, hidd, cont
    
seed_seq = 'A cheap alternative to'
generated_seq = seed_seq

checkpoint = torch.load(r"C:\Users\HP\Desktop\Piyush\Languages and Frameworks\Python\NLPusingDL\mp3\best_model", map_location=torch.device('cpu'))

hidd = torch.zeros(2, 1, 200)
cont = torch.zeros(2, 1, 200)
model = LSTM()
model.load_state_dict(checkpoint['model'])

seed_to_int = [vocab[i] for i in list(seed_seq)]
initial_seed_tensor = torch.tensor(seed_to_int)
out, hidd, cont = model(initial_seed_tensor, hidd, cont)

for i in range(200):
    x = generated_seq[len(generated_seq)-1]
    x_tensor = torch.tensor([vocab[x]])
    out, hidd, cont = model(x_tensor, hidd, cont)
    out = out.squeeze(0)
    out = out.squeeze(0)
    out[out < 0.0] = 0
    generated_char = torch.multinomial(out, 1).item()
    generated_char = i_to_word[generated_char]
    generated_seq += generated_char

print(generated_seq)
