# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastText(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout, n_class=2):
        super(FastText, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.n_class = n_class
        self.max_num = 800
        # self.embedding = nn.Embedding(self.max_words, self.emb_size)
        # self.embedding_ngram2 = nn.Embedding(self.max_words, self.emb_size)
        # self.embedding_ngram3 = nn.Embedding(self.max_words, self.emb_size)
        # self.dropout2 = nn.Dropout(self.dropout)
        # self.fc1 = nn.Linear(self.emb_size * 3, self.hid_size)
        # self.fc2 = nn.Linear(self.hid_size, 2)
        super(FastText, self).__init__()
        self.embed = nn.Embedding(self.max_words, self.emb_size)
        self.avg_pool = nn.MaxPool1d(kernel_size=self.max_num, stride=1)
        self.fc = nn.Linear(self.emb_size, self.n_class)

    def forward(self, x):
        # out_word = self.embedding(x[0])
        # out_bigram = self.embedding_ngram2(x[2])
        # out_trigram = self.embedding_ngram3(x[3])
        # out = torch.cat((out_word, out_bigram, out_trigram), -1)
        # out = out.mean(dim=1)
        # out = self.dropout2(out)
        # out = self.fc1(out)
        # out = F.relu(out)
        # out = self.fc2(out)
        # return outx = torch.Tensor(x).long()  # b, max_len
        x = self.embed(x)  # b, max_len, embedding_dim
        x = x.transpose(2, 1).contiguous()  # b, embedding_dim, max_len
        x = self.avg_pool(x).squeeze()  # b, embedding_dim, 1
        out = self.fc(x)  # b, num_label
        return out
