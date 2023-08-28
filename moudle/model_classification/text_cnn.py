# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

Config = {
    "conv1D_out": [8, 8, 8],  # 1D-conv层的output-channel列表
    "conv1D_ker": [2, 3, 4],  # 1D-conv层的kernel尺寸列表
}


class TextCnnModel(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout, n_class):
        super(TextCnnModel, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.n_class = n_class
        self.embedding = nn.Embedding(self.max_words, self.emb_size)
        self.conv1D = nn.ModuleList([nn.Conv1d(in_channels=self.emb_size, out_channels=out, kernel_size=ker)
                                     for out, ker in zip(Config['conv1D_out'], Config['conv1D_ker'])])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(Config['conv1D_out']), self.n_class)

    def forward(self, x):
        x = self.embedding(x)  # x: (batch, sequence, embed)
        x = x.permute(0, 2, 1)  # x :(batch, embed, sequence)  将embed视为in_channel，这样才能进行1维卷积
        x = [F.relu(conv1D(x)) for conv1D in self.conv1D]  # [(batch, out_channel, L_out)]
        x = [F.max_pool1d(i, i.size(-1)) for i in x]  # [(batch, out_channel, 1)]，在最后一个维度上进行max_pooling
        x = [torch.squeeze(i, dim=-1) for i in x]  # [(batch, out_channel)]，维度压缩
        x = torch.cat(x, dim=-1)  # (batch, total_out_channel), 沿着各out_channel进行拼接
        x = self.dropout(x)
        out = self.fc(x)
        return out
