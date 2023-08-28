# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout, n_class=2):
        super(LSTMModel, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.n_class = n_class
        self.embedding = nn.Embedding(self.max_words, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hid_size, num_layers=2, batch_first=True, bidirectional=True)
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.n_class)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.embedding(x)
        x = self.dp(x)
        x, _ = self.lstm(x)
        x = self.dp(x)
        x = F.relu(self.fc1(x))
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
        out = self.fc2(x)
        return out
