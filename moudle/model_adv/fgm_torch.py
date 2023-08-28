# -*- coding:utf-8 -*-
import torch


class FGM(object):
    def __init__(self, model, emb_name):
        self.model = model
        self.backup = {}
        self.emb_name = emb_name

    def check(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                print("已经找到该层，可以进行绕动")
                print(name)
                return True
        return False

    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
                # print("已经执行restore")
        self.backup = {}

