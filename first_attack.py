# -*- coding:utf-8 -*-
import pickle
import random
from collections import Counter

import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# from model_attack_method import path_shuffle
# from pgd_file import pad_tensor  # pgd
# from utils import record_important_layer, store_file  # 记录重要程度
# from fgm_file import pad_tensor  # fgm
# from attck_model_none import pad_tensor  # 对抗攻击--推理
from attack_model_pgd import pad_pgd_tensor  # 对抗攻击--推理-pgd
from attack_model_fgm import pad_fgm_tensor  # 对抗攻击--推理-fgm
# from file_none import pad_tensor  # 无对抗训练
from url_attack import params_attack
from url_attack import components_fragment_attack
from url_attack import components_fragment_second_attack
from url_attack import components_query_attack
from url_attack import components_query_second_attack
from url_attack import components_params_attack
from url_attack import components_params_second_attack
from url_attack import components_path_attack
from url_attack import components_test_attack
from model_attack_method import character_level
from url_attack import components_netloc_attack
from url_attack import components_scheme_attack
from url_attack import total_params_attack
import swifter

pf = pd.read_csv("./datafile/phishing_site_urls.csv")
print(pf[:4])

tokenizer = pickle.load(open("./model/tokenizer.pkl", "rb"))
print("first")
# print(len(tokenizer.word_index) + 1)
# exit()
# 记录重要度
record_list = []


# 字符级绕动
# def character_level(url: str):
#     url = url.replace("[", "").replace("]", "")
#     # 替换重要字符  第三种方法
#     # 替换末尾字符（不重要字符长度）
#     # statistic = Counter(url).most_common()
#     # # last = statistic[-1]
#     # level_import = 0
#     # len_url = len(statistic)
#     # if len_url >= level_import:
#     #     last = statistic[-level_import]
#     # else:
#     #     last = statistic[-len_url]
#     # index_ch = ord(last[0])
#     # conditional = [chr(_) for _ in range(65, 91) if _ != index_ch] + [str(_) for _ in range(0, 10) if _ != index_ch] + [chr(_) for _ in range(97, 123) if _ != index_ch]
#     # table = str.maketrans(last[0], random.choice(conditional))
#     # url = url.translate(table)
#     # 固定替换字符、对比对抗训练（最多+最少）=====
#     record = Counter(url)
#     statistic = record.most_common()
#     statistic = list(filter(lambda x: 65 <= ord(x[0]) <= 90 or 48 <= ord(x[0]) <= 57 or 97 <= ord(x[0]) <= 122, statistic))  # 保留. \ ? =
#     # last = statistic[-1]
#     get_ch = 1
#     len_url = len(statistic)
#     if len_url >= get_ch:
#         last = [statistic[0]]  # 这里控制位置
#     else:
#         last = statistic[-len_url:]
#     index_ch = [ord(_[0]) for _ in last]
#     # record_list.extend(record_important_layer(record, [_[0] for _ in last], len(url)))  # 启用记录函数
#     conditional = [chr(_) for _ in range(65, 91) if _ not in index_ch] + [str(_) for _ in range(0, 10) if _ not in index_ch] + [chr(_) for _ in range(97, 123) if _ not in index_ch]
#     table = str.maketrans(''.join([_[0] for _ in last]), ''.join(random.sample(conditional, k=len(last))))
#     url = url.translate(table)
#     # 随机替换字符  第二种方法 =====
#     # reflect_dict = Counter(url)
#     # str_set = set(url)
#     # # str_set = list(filter(lambda x: 65 <= ord(x[0]) <= 90 or 48 <= ord(x[0]) <= 57 or 97 <= ord(x[0]) <= 122, str_set))  # 保留. \ ? =
#     # get_ch = 3
#     # # print(f"随机{get_ch}类")
#     # if len(str_set) < get_ch:
#     #     get_ch = len(str_set)
#     # chr_ch = random.sample(str_set, k=get_ch)
#     # # record_list.extend(record_important_layer(reflect_dict, chr_ch, len(url)))  # 启用记录函数
#     # index_ch = list(map(lambda x: ord(x), chr_ch)) + [46, 47, 63, 38, 61]  # 依次表示为. / ? & =
#     # conditional = [chr(_) for _ in range(65, 91) if _ not in index_ch] + [str(_) for _ in range(0, 10) if _ not in index_ch] + [chr(_) for _ in range(97, 123) if _ not in index_ch]
#     # re_ch = random.sample(conditional, k=get_ch)
#     # table = str.maketrans(''.join(chr_ch), ''.join(re_ch))
#     # url = url.translate(table)
#     # 旧方法url.replace(chr_ch, random.choice(conditional))
#     # 修改域名  第一种方法
#     # url = url.replace("com", 'cn')
#     return url

# 声明标签数据编码器
labels = pf.Label
label_coder = preprocessing.LabelEncoder()
# 训练
label_coder.fit(labels)
# 对应表
# print(label_coder.classes_)
print(label_coder.transform(["bad", "good"]))
# 转成数字
result_tr = label_coder.transform(labels)
# one-hot
labels = to_categorical(result_tr)

TRAIN_SIZE = 0.8

X_train, X_test, y_train, y_test = train_test_split(pf, result_tr, test_size=0.2, random_state=42)


attack_flag = True
if attack_flag:
    # 攻击开始
    print("替换开始")
    # print('*' * 25)
    # print("替换前的结果")
    for i in X_test.URL[:10]:
        print(i.lower())
    # pf.URL = pf.URL.apply(path_shuffle)
    # ==================================== url structure  =====================
    # pf.URL = pf.URL.apply(params_attack)
    # =========================================================================
    # ********************************url path**************************
    # X_test.URL = X_test.URL.swifter.apply(components_path_attack)
    # print("detail finish")
    # ***********************************************************************
    # ********************************url query**************************
    # pf.URL = pf.URL.swifter.apply(components_query_attack)  # 0.7207, 效果更好，其他的模型这个方式更好
    # X_test.URL = X_test.URL.swifter.apply(components_query_second_attack)  # LSTM+Attention、CNN有效
    # print("detail finish")
    # ***********************************************************************
    # ********************************url fragment**************************
    # 两个方法差异不大
    # pf.URL = pf.URL.apply(components_fragment_attack)  # 0.9349
    # print("detail finish")
    # X_test.URL = X_test.URL.swifter.apply(components_fragment_second_attack)  # 效果很好
    # ***********************************************************************
    # ********************************url netloc **************************
    # X_test.URL = X_test.URL.swifter.apply(components_netloc_attack)
    # print("detail finish")
    # ***********************************************************************
    # ********************************url params **************************
    # X_test.URL = X_test.URL.swifter.apply(components_params_attack)
    # print("detail finish")
    # 这个方法在数据集1无用pf.URL = pf.URL.apply(components_params_second_attack)
    # ***********************************************************************
    # ********************************url scheme **************************
    # X_test.URL = X_test.URL.swifter.apply(components_scheme_attack)
    # print("detail finish")
    # ***********************************************************************
    # ********************************url total_params **************************
    # pf.URL = X_test.URL.swifter.apply(total_params_attack)  # 总参数
    # print("finish")
    # ***********************************************************************
    # ********************************url level **************************
    # X_test.URL = X_test.URL.swifter.apply(character_level, args=(True, ))  # scheme用
    X_test.URL = X_test.URL.apply(character_level, args=(False, ))  # 无scheme的时候用
    # ***********************************************************************
    # ******************************** test **************************
    # pf.URL = pf.URL.apply(components_test_attack)
    # ***********************************************************************
    # 攻击结束
    # print("-" * 50)
    # print("替换后的结果")
    for i in X_test.URL[:10]:
        print(i.lower())
    print("替换结束")
    file_name = "./result/string_ratio/first_data_two_FS_class.txt"
    # store_file(record_list, file_name)
    exit()
    # 攻击结束
sequences = tokenizer.texts_to_sequences(X_test.URL)

MAX_NUM_WORDS = 800
# fast text下的数据维度不一致，因此将其改成384
# MAX_NUM_WORDS = 384
X_test = pad_sequences(sequences, maxlen=MAX_NUM_WORDS, padding="post")


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# pad_tensor(X_train, y_train, X_test, y_test, len(label_coder.classes_), "first")
pad_pgd_tensor(X_train, y_train, X_test, y_test, len(label_coder.classes_), "first")
pad_fgm_tensor(X_train, y_train, X_test, y_test, len(label_coder.classes_), "first")


# [[30817   383]
#  [13583 65087]]


# [[29682  1518]
#  [ 1846 76824]]

# [[11693 19507]
#  [ 2481 76189]]


