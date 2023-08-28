# -*- coding:utf-8 -*-
import time

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import RandomSampler

from module.model_classification.lstm_model import LSTMModel  # LSTM模型
from module.model_classification.text_r_cnn import TextRCnnModel  # TextRCNN模型
from module.model_classification.rnn_att import RNNAttention  # RNNAttention
from module.model_classification.text_cnn import TextCnnModel  # TextCNN模型
from module.model_classification.fast_text import FastText  # FastText模型


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

print(DEVICE)

torch.backends.cudnn.deterministic = True

loss_file = "./result/train_history/second/loss_distribution_pgd.txt"
# 下面的是第一个数据集的参数
record_point = 10  # 控制记录的个数 对于第二个数据集来说需要设计的小一点
BATCH_SIZE = 256  # 这个两个决定记录点的个数

MAX_WORDS = 487712
# MAX_LEN = 350
EMB_SIZE = 128
HID_SIZE = 128
DROPOUT = 0.2

model_dict = {
    "fasttext": FastText,
    "lstm": LSTMModel,
    'rnnatt': RNNAttention,
    "textcnn": TextCnnModel,
    "rcnn": TextRCnnModel,
}


def pad_tensor(x_train, y_train, x_test, y_test, n_class, file_name):
    print("attack")
    # train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    # train_sampler = RandomSampler(train_data)
    # train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    for name, model_type in model_dict.items():
        print(f"当前模型:{name}")
        test_label = start_train(model_type, test_loader, file_name, n_class=n_class)
        print(test_label.count(1))
        print(test_label.count(0))
        cfg = confusion_matrix(y_test, test_label)
        print(cfg)
        tn, fp, fn, tp = cfg.ravel()
        print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
        print(f"{name} malicious acc:{tn/(tn+fp):.4f}")
        print("===================================================")
    # print(tp/(fn+tp))
    # with open(loss_file, "w") as f:
    #     f.writelines(list(map(lambda x: str(x) + '\n', loss_dis)))


def start_train(model_type, test_loader, file_name, n_class):
    model = model_type(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT, n_class=n_class).to(DEVICE)
    print(model)
    PATH = "./model_save/" + str(file_name) + "/" + model.__class__.__name__ + ".pth"
    print(PATH)
    best_pre = []
    loss_pre = []
    # optimizer = optim.Adam(model.parameters())
    # start = time.clock()
    # model.load_state_dict(torch.load(PATH), strict=False)
    model = torch.load(PATH)
    # model1 = torch.nn.DataParallel(model, device_ids=[0, 1])  # 多GPU使用
    test_result = re_test(model, DEVICE, test_loader)
    acc = test_result[0]
    test_label = test_result[1]
    test_loss = test_result[2]
    best_pre = test_label
    print("acc is: {:.4f}".format(acc))
    return best_pre


def re_test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    test_loss = 0.0
    acc = 0
    pred_label = []
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            y_ = model(x)
            print(y_)
        exit()
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]
        pred_label.extend(pred.view(-1).detach().cpu().numpy())
        acc += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss:{:.4f}, Accuracy: {}/{} ({:.2f}%)".format(test_loss, acc, len(test_loader.dataset), 100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset), pred_label, test_loss
