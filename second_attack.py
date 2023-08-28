# -*- coding:utf-8 -*-
import pickle

import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# from fgm_file import pad_tensor  # fgm
# from attck_model_none import pad_tensor  # 对抗攻击--推理
from attack_model_pgd import pad_pgd_tensor  # 对抗攻击--推理-pgd
from attack_model_fgm import pad_fgm_tensor  # 对抗攻击--推理-fgm
# from file_none import pad_tensor  # 无对抗训练
# from train_model_test import pad_tensor
# from model_attack_method import attack_path
# from pgd_file import pad_tensor  # pgd
# from utils import record_important_layer, store_file  # 记录重要程度
# from model_attack_method import port_attack
from model_attack_method import character_level
from model_attack_method import store_start
from url_attack import params_attack
from url_attack import components_fragment_attack
from url_attack import components_fragment_second_attack
from url_attack import components_query_attack
from url_attack import components_query_second_attack
from url_attack import components_params_attack
from url_attack import components_params_second_attack
from url_attack import components_path_attack
from url_attack import components_path_second_attack
from url_attack import components_test_attack
from url_attack import components_netloc_attack
from url_attack import components_scheme_attack
import swifter

pf = pd.read_csv("./datafile/combined_dataset.csv")
print(pf[:4])

tokenizer = pickle.load(open("./model/tokenizer.pkl", "rb"))
print("second")
# print(pf.label.value_counts())


# print(len(tokenizer.word_index) + 1)
# exit()
# 记录重要度
record_list = []

# 字符级绕动
# def character_level(url: str):
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
#     # record = Counter(url)
#     # statistic = record.most_common()
#     # statistic = list(filter(lambda x: 65 <= ord(x[0]) <= 90 or 48 <= ord(x[0]) <= 57 or 97 <= ord(x[0]) <= 122, statistic))  # 保留. \ ? =
#     # # last = statistic[-1]
#     # get_ch = 3
#     # len_url = len(statistic)
#     # if len_url >= get_ch:
#     #     last = [statistic[0]] + [statistic[-1]] + [statistic[1]]  # 这里控制位置
#     # else:
#     #     last = statistic[-len_url:]
#     # index_ch = [ord(_[0]) for _ in last]
#     # record_list.extend(record_important_layer(record, [_[0] for _ in last], len(url)))  # 启用记录函数
#     # conditional = [chr(_) for _ in range(65, 91) if _ not in index_ch] + [str(_) for _ in range(0, 10) if _ not in index_ch] + [chr(_) for _ in range(97, 123) if _ not in index_ch]
#     # table = str.maketrans(''.join([_[0] for _ in last]), ''.join(random.sample(conditional, k=len(last))))
#     # url = url.translate(table)
#     # 随机替换字符  第二种方法 =====
#     reflect_dict = Counter(url)
#     str_set = set(url)
#     # str_set = list(filter(lambda x: 65 <= ord(x[0]) <= 90 or 48 <= ord(x[0]) <= 57 or 97 <= ord(x[0]) <= 122, str_set))  # 保留. \ ? =
#     get_ch = 3
#     # print(f"随机{get_ch}类")
#     if len(str_set) < get_ch:
#         get_ch = len(str_set)
#     chr_ch = random.sample(str_set, k=get_ch)
#     record_list.extend(record_important_layer(reflect_dict, chr_ch, len(url)))  # 启用记录函数
#     index_ch = list(map(lambda x: ord(x), chr_ch)) + [46, 47, 63, 38, 61]  # 依次表示为. / ? & =
#     conditional = [chr(_) for _ in range(65, 91) if _ not in index_ch] + [str(_) for _ in range(0, 10) if _ not in index_ch] + [chr(_) for _ in range(97, 123) if _ not in index_ch]
#     re_ch = random.sample(conditional, k=get_ch)
#     table = str.maketrans(''.join(chr_ch), ''.join(re_ch))
#     url = url.translate(table)
#     # 旧方法url.replace(chr_ch, random.choice(conditional))
#     # 修改域名  第一种方法
#     # url = url.replace("com", 'cn')
#     return url


attack_flag = True
if attack_flag:
    # 攻击开始
    print("替换开始")
    # print('*' * 25)
    # print("替换前的结果")
    # print(pf.domain[:4])
    for i in pf.domain[:10]:
        print(i)
    # *********************************端口攻击*********************************
    # pf.domain = pf.domain[:2].apply(port_attack)
    # print(pf.domain[:2])
    # exit()
    # ************************************************************************
    # *******************************域名替换*********************************
    # with open(r'./adversarialSampleAnalyse/statistic_domain.txt', 'r') as f:
    #     instead_url = [i.strip('\n') for i in f.readlines()]
    # pf.domain = pf.apply(domain_attack, axis=1, args=instead_url)
    # ************************************************************************
    # *******************************路径攻击********************************
    # pf.domain = pf.domain.apply(attack_path)
    # ***********************************************************************
    # *******************************路径删除攻击********************************
    # pf.domain = pf.domain.apply(character_delete)
    # ***********************************************************************
    # *******************************路径增加攻击********************************
    # pf.domain = pf.domain.apply(character_additional)
    # ***********************************************************************
    # *******************************路径重排攻击********************************
    # pf.domain = pf.domain.apply(path_shuffle)
    # ***********************************************************************
    # *******************************片段攻击********************************
    # pf.domain = pf.domain.apply(segment_attack)
    # ***********************************************************************
    # ********************************url structure**************************
    # pf.domain = pf.domain.apply(params_attack)
    # ***********************************************************************
    # ********************************url path**************************
    # pf.domain = pf.domain.apply(components_path_attack)  # textcnn和lstmattention不一样效果更好
    pf.domain = pf.domain.swifter.apply(components_path_second_attack)  # 效果更好,  组合攻击的时候效果更好
    # ***********************************************************************
    # ********************************url query**************************
    # pf.domain = pf.domain.apply(components_query_attack)
    pf.domain = pf.domain.swifter.apply(components_query_second_attack)  # 有效
    # ***********************************************************************
    # ********************************url fragment**************************
    pf.domain = pf.domain.swifter.apply(components_fragment_second_attack)  # 效果更好
    # pf.domain = pf.domain.apply(components_fragment_attack)
    # ***********************************************************************
    # ********************************url netloc**************************
    pf.domain = pf.domain.swifter.apply(components_netloc_attack)
    # ***********************************************************************
    # ********************************url params **************************
    # pf.domain = pf.domain.apply(components_params_attack)  # 不采取这个方法
    pf.domain = pf.domain.swifter.apply(components_params_second_attack)  # 构建的效果很好
    # ***********************************************************************
    # ********************************url scheme **************************
    pf.domain = pf.domain.swifter.apply(components_scheme_attack)
    # ***********************************************************************
    # ********************************url test **************************
    # pf.domain = pf.domain.apply(components_test_attack)
    # ***********************************************************************
    # *********************************随机替换*********************************
    num = 14
    pf.domain = pf.domain.swifter.apply(character_level, args=(num, True))  # scheme用
    # pf.domain = pf.domain.swifter.apply(character_level, args=(num, False))  # 无scheme用
    # ************************************************************************
    # 攻击结束
    # print("-" * 50)
    # print("替换后的结果")
    # print(pf.domain[:4])
    for i in pf.domain[:10]:
        print(i)
    print("替换结束")
    # file_name = "./result/string_ratio/second_data_random_3_class.txt"
    # file_name = f"./result/append_result/second/second_data_{num}_class.txt"
    # store_file(record_list, file_name)
    # store_start(file_name)
    exit()
    # 攻击结束
# exit()
sequences = tokenizer.texts_to_sequences(pf.domain)
# 第二个数据集调一个大一点的参数
MAX_NUM_WORDS = 800
# MAX_NUM_WORDS = 350
# fast text下的数据维度不一致，因此将其改成384
# MAX_NUM_WORDS = 384
data = pad_sequences(sequences, maxlen=MAX_NUM_WORDS, padding="post")
reflect = {0: "good", 1: "bad"}  # 为了和后面的结果统一
pf.label = pf.label.apply(lambda x: reflect[x])
# 声明标签数据编码器
labels = pf.label
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
X_train, X_test, y_train, y_test = train_test_split(data, result_tr, test_size=0.2, random_state=42)
# print(type(X_test))
# exit()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# pad_tensor(X_train, y_train, X_test, y_test, len(label_coder.classes_), "second")
pad_pgd_tensor(X_train, y_train, X_test, y_test, len(label_coder.classes_), "second")
pad_fgm_tensor(X_train, y_train, X_test, y_test, len(label_coder.classes_), "second")

# [[10475   897]
#  [ 3835  3975]]
# tn:10475, fp:897, fn:3835, tp:3975

# [[11144   228]
#  [  260  7550]]
# tn:11144, fp:228, fn:260, tp:7550

# Netloc+path+fragment-level-10的结果由下面的几个组成:components_path_second_attack、components_fragment_attack、components_netloc_attack、character_level
