import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
from constants import *

class LSTMNN(nn.Module):
    def __init__(self):
        super().__init__()
        w2vmodel = gensim.models.KeyedVectors.load('./models/' + 'word2vec_'+str(EMBEDDING_SIZE)+'_PAD.model')
        weights = w2vmodel.wv
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=w2vmodel.wv.key_to_index['pad'])
        self.lstm=nn.LSTM(EMBEDDING_SIZE,HIDDEN_SIZE,batch_first=True,num_layers=2)
        self.fc=nn.Linear(HIDDEN_SIZE,NUM_CLASSES)
    def forward(self,x):
        x=self.embedding(x)
        x=self.lstm(x)
        return self.fc(x[1][0][-1])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        w2vmodel = gensim.models.KeyedVectors.load('./models/' + 'word2vec_'+str(EMBEDDING_SIZE)+'_PAD.model')
        weights = w2vmodel.wv
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=w2vmodel.wv.key_to_index['pad'])
        window_sizes=(1,2,3,5,10)
        self.convs = nn.ModuleList([
                                   nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0))
                                   for window_size in window_sizes
        ])

        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x) # [B, T, E]

        # Apply a convolution + max_pool layer for each window size
        x = torch.unsqueeze(x, 1)
        xs = []
        for conv in self.convs:
            x2 = F.relu6(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)

        # FC
        x = x.view(x.size(0), -1)
        probs = self.fc(x)

        return probs
    
if __name__ == "__main__":
    pass