# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAttention(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout, n_class):
        super(RNNAttention, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.n_class = n_class
        self.embedding = nn.Embedding(self.max_words, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hid_size, num_layers=2, batch_first=True, bidirectional=True)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(self.hid_size * 2))
        self.fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc = nn.Linear(self.hid_size, self.n_class)

    def forward(self, x):
        self.lstm.flatten_parameters()
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[256, 350, 350]
        h, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
        m = self.tanh1(h)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(m, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = h * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
