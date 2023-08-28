# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return torch.max_pool1d(x, kernel_size=x.shape[-1])


class TextRCnnModel(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout, n_class):
        super(TextRCnnModel, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.n_class = n_class
        self.embedding = nn.Embedding(self.max_words, self.emb_size, padding_idx=self.max_words - 1)
        self.lstm = nn.LSTM(self.emb_size, self.hid_size, num_layers=2, batch_first=True, bidirectional=True)
        # self.maxpool_t = nn.MaxPool1d(lambda x: x)
        self.maxpool = GlobalMaxPool1d()
        self.dropout_l = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hid_size * 2 + self.emb_size, 256)
        self.fc1 = nn.Linear(256, self.n_class)

    def forward(self, x):
        self.lstm.flatten_parameters()
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]  256, 350, 350
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(self.fc(out))
        out = out.permute(dims=[0, 2, 1])
        out = self.maxpool(out).squeeze(-1)
        out = self.dropout_l(out)
        out = self.fc1(out)
        return out
