# -*- coding:utf-8 -*-
import time

import torch
import torch.nn as nn
import torch.optim as optim
# from module.model_classification.text_r_cnn import TextRCnnModel  # TextRCNN模型
# from module.model_classification.rnn_att import RNNAttention  # RNNAttention
from module.model_classification.text_cnn import TextCnnModel  # TextCNN模型
# from module.model_classification.fast_text import FastText  # FastText模型
# from module.model_classification.transform_model import TransModel  # Transform模型
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import RandomSampler

from module.model_adv.pgd_torch import PGD  # 对抗模型
from torchtools import EarlyStopping
# from module.model_classification.lstm_model import LSTMModel  # LSTM模型

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

print(DEVICE)

torch.backends.cudnn.deterministic = True

loss_file = "./result/train_history/second/loss_distribution_pgd.txt"
# 下面的是第一个数据集的参数
record_point = 10  # 控制记录的个数 对于第二个数据集来说需要设计的小一点
BATCH_SIZE = 256  # 这个两个决定记录点的个数


# 下面的是第二个数据集的参数
# record_point = 5  # 控制记录的个数 对于第二个数据集来说需要设计的小一点
# BATCH_SIZE = 128  # 这个两个决定记录点的个数


def pad_tensor(x_train, y_train, x_test, y_test, n_class, file_num):
    print("pgd")
    train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    test_label, loss_dis = start_train(train_loader, test_loader, n_class=n_class, file_num=file_num)
    cfg = confusion_matrix(y_test, test_label)
    print(cfg)
    tn, fp, fn, tp = cfg.ravel()
    print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
    with open(loss_file, "w") as f:
        f.writelines(list(map(lambda x: str(x) + '\n', loss_dis)))


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    pgd = PGD(model, "embedding", epsilon=17.0, alpha=2.0)
    if pgd.check():
        print("检测成功！")
    # 实验结果中不标注K的默认为3
    K = 3
    loss_store = []
    train_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        # 正常训练
        y_ = model(x)
        loss = criterion(y_, y)
        train_loss += loss
        loss.backward()
        # 对抗训练部分
        pgd.backup_grad()
        for t in range(K):
            pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K - 1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            loss_adv_t = model(x)
            loss_adv = criterion(loss_adv_t, y)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            loss_store.append(round(loss_adv.item(), 4))
        pgd.restore()
        # 结束位置
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
        if (batch_idx + 1) % record_point == 0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}".format(epoch, batch_idx * len(x), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    return loss_store, train_loss


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
        test_loss += criterion(y_, y)
        pred = y_.max(-1, keepdim=True)[1]
        pred_label.extend(pred.view(-1).detach().cpu().numpy())
        acc += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss:{:.4f}, Accuracy: {}/{} ({:.2f}%)".format(test_loss, acc, len(test_loader.dataset), 100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset), pred_label, test_loss


MAX_WORDS = 487712
# MAX_LEN = 350
EMB_SIZE = 128
HID_SIZE = 128
DROPOUT = 0.2


def start_train(train_loader, test_loader, n_class, file_num):
    model = TextCnnModel(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT, n_class=n_class).to(DEVICE)
    print(model)
    # model = torch.nn.DataParallel(model1, device_ids=[0, 1])  # 多GPU使用
    optimizer = optim.Adam(model.parameters())
    best_acc = 0.0
    # PATH = "./result/model/model.pth"
    PATH = "./model_save/adversarial_attack/pgd/" + str(file_num) + "/" + model.__class__.__name__ + ".pth"
    print(PATH)
    best_pre = []
    loss_pre = []
    test_loss_s = []
    train_loss_s = []
    test_acc_s = []
    patience = 3
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    test_loss_file = "./result/train_history/loss_epoch/loss_test_pgd(text_cnn).txt"
    train_loss_file = "./result/train_history/loss_epoch/loss_train_pgd(text_cnn).txt"
    test_acc_file = "./result/train_history/loss_epoch/test_acc_pgd(text_cnn).txt"
    start = time.clock()
    for epoch in range(1, 51):
        train_result = train(model, DEVICE, train_loader, optimizer, epoch)
        loss_dis = train_result[0]
        train_loss = train_result[1]
        test_result = re_test(model, DEVICE, test_loader)
        acc = test_result[0]
        test_label = test_result[1]
        test_loss = test_result[2]
        test_loss_s.append(str(test_loss) + '\n')
        train_loss_s.append(str(train_loss) + '\n')
        test_acc_s.append(str(round(acc, 4)) + "\n")
        if best_acc < acc:
            best_acc = acc
            # torch.save(model.state_dict(), PATH)
            torch.save(model, PATH)
            print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))
            best_pre = test_label
        loss_pre.extend(loss_dis)
        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    end = time.clock()
    print(f"运行时间{end - start}s")
    with open(test_loss_file, 'w')as f_te_l, open(train_loss_file, 'w') as f_tr_l, open(test_acc_file, 'w') as f_te_acc:
        f_te_l.writelines(test_loss_s)
        f_tr_l.writelines(train_loss_s)
        f_te_acc.writelines(test_acc_s)
    return best_pre, loss_pre
