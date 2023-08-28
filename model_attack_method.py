# -*- coding:utf-8 -*-
# 域名替换攻击
import random
import re
from collections import Counter
from url_attack import get_url_part
from url_attack import get_url_part_special
from urllib.parse import urlunparse
from utils import record_important_layer, store_file  #记录重要程度


def domain_attack(df_iter, *conditional_domain):
    domain_url = df_iter['domain'][:df_iter['domainLen']][::-1]
    if '.' not in domain_url:
        return domain_url[::-1]
    split_position = domain_url.index('.')
    if split_position == 0:
        split_position = domain_url[1:].index('.')
        get_url = domain_url[1:][:split_position][::-1]
    else:
        get_url = domain_url[:split_position][::-1]
    result = re.findall(r'[a-zA-Z]+', get_url)
    if result:
        filter_result = list(filter(lambda x: x != result[0], conditional_domain))
        get_one = random.choice(filter_result)
        new_url = domain_url[split_position:][::-1] + get_one + df_iter['domain'][df_iter['domainLen']:]
        return new_url
    return domain_url[split_position:][::-1] + random.choice(conditional_domain) + df_iter['domain'][df_iter['domainLen']:]


def store_start(file_name):
    store_file(record_list, file_name=file_name)


record_list = []  # 用于记录重要程度


# 字符级绕动
def character_level(url: str, number, scheme=False):
    # 替换重要字符  第三种方法
    # 替换末尾字符（不重要字符长度）
    # statistic = Counter(url).most_common()
    # # last = statistic[-1]
    # level_import = 0
    # len_url = len(statistic)
    # if len_url >= level_import:
    #     last = statistic[-level_import]
    # else:
    #     last = statistic[-len_url]
    # index_ch = ord(last[0])
    # conditional = [chr(_) for _ in range(65, 91) if _ != index_ch] + [str(_) for _ in range(0, 10) if _ != index_ch] + [chr(_) for _ in range(97, 123) if _ != index_ch]
    # table = str.maketrans(last[0], random.choice(conditional))
    # url = url.translate(table)
    # 替换字符（最多+最少）
    record = Counter(url)
    statistic = record.most_common()
    statistic = list(filter(lambda x: 65 <= ord(x[0]) <= 90 or 48 <= ord(x[0]) <= 57 or 97 <= ord(x[0]) <= 122, statistic))  # 保留. \ ? =
    # last = statistic[-1]
    get_ch = number  # 需要替换的长度
    len_url = len(statistic)  # 字符总长度
    if len_url >= get_ch:
        #  + [statistic[2]] + [statistic[1]] + [statistic[3]] + [statistic[4]] + [statistic[5]]
        # last = [statistic[0]] + [statistic[1]] + [statistic[2]] + [statistic[3]]  # 这里控制位置
        last = []
        for i in range(get_ch):
            last.extend([statistic[i]])
    else:
        last = statistic[-len_url:]
    # index_ch是满足字符的要求
    index_ch = [ord(_[0]) for _ in last] + [46, 47, 63, 38, 61]  # 依次表示为. / ? & =
    # conditional选择一些满足要求的字符
    conditional = [chr(_) for _ in range(65, 91) if _ not in index_ch] + [str(_) for _ in range(0, 10) if _ not in index_ch] + [chr(_) for _ in range(97, 123) if _ not in index_ch]
    # 构建转化表
    table = str.maketrans(''.join([_[0] for _ in last]), ''.join(random.sample(conditional, k=len(last))))
    record_list.extend(record_important_layer(record, [_[0] for _ in last], len(url)))
    if not scheme:
        result = get_url_part(url)
    else:
        result = get_url_part_special(url)
    new_path = result.path.translate(table)
    new_fragment = result.fragment.translate(table)
    new_param = result.params.translate(table)
    new_query = result.query.translate(table)
    # url = url.translate(table)
    if not scheme:
        new_url = result._replace(scheme="", path=new_path, params=new_param, query=new_query, fragment=new_fragment)
        return urlunparse(new_url)[2:]
    else:
        new_url = result._replace(path=new_path, params=new_param, query=new_query, fragment=new_fragment)
        return urlunparse(new_url)
    # 随机替换字符  第二种方法
    # str_set = set(url)
    # get_ch = 3
    # if len(str_set) < get_ch:
    #     get_ch = len(str_set)
    # chr_ch = random.sample(str_set, k=get_ch)
    # index_ch = list(map(lambda x: ord(x), chr_ch)) + [46, 47, 63, 38, 61]  # 依次表示为. / ? & =
    # conditional = [chr(_) for _ in range(65, 91) if _ not in index_ch] + [str(_) for _ in range(0, 10) if _ not in index_ch] + [chr(_) for _ in range(97, 123) if _ not in index_ch]
    # re_ch = random.sample(conditional, k=get_ch)
    # table = str.maketrans(''.join(chr_ch), ''.join(re_ch))
    # url = url.translate(table)
    # 旧方法url.replace(chr_ch, random.choice(conditional))
    # 修改域名  第一种方法
    # url = url.replace("com", 'cn')

    # return url


def attack_path(domain: str):
    get_result = re.findall(r"/*(.*?)/", domain)
    if not get_result:
        return domain
    index_get = domain[::-1].find(get_result[-1][::-1]) - 1
    tail = domain[::-1][:index_get][::-1]
    kernel = get_result[1:]
    random.shuffle(kernel)
    new_url_list = [get_result[0]] + kernel + [tail]
    return '/'.join(new_url_list)


def character_delete(domain: str):
    get_result = re.findall(r"/*(.*?)/", domain)
    if not get_result:
        return domain
    index_get = domain[::-1].find(get_result[-1][::-1]) - 1
    tail = domain[::-1][:index_get][::-1]
    kernel = get_result[1:]
    if len(kernel) >= 2:
        kernel.remove(random.choice(kernel))
        kernel.remove(random.choice(kernel))
    new_url_list = [get_result[0]] + kernel + [tail]
    return '/'.join(new_url_list)


def character_additional(domain: str):
    get_result = re.findall(r"/*(.*?)/", domain)
    if not get_result:
        return domain
    index_get = domain[::-1].find(get_result[-1][::-1]) - 1
    tail = domain[::-1][:index_get][::-1]
    kernel = get_result[1:]
    if kernel:
        kernel.extend([random.choice(kernel)])
    random.shuffle(kernel)
    char_list = [i for i in range(48, 58)] + [i for i in range(65, 91)]
    new_url_list = [get_result[0]] + [i + chr(random.choice(char_list)) for i in kernel] + [tail]
    return '/'.join(new_url_list)


def path_shuffle(domain: str):
    get_result = re.findall(r"/*(.*?)/", domain)
    if not get_result:
        return domain
    index_get = domain[::-1].find(get_result[-1][::-1]) - 1
    tail = domain[::-1][:index_get][::-1]
    kernel = get_result[1:]
    if kernel:
        kernel.extend([random.choice(kernel)])
    random.shuffle(kernel)
    shuffle_kernel = []
    for i in kernel:
        str_one = list(i)
        random.shuffle(str_one)
        shuffle_kernel.append(''.join(str_one))
    new_url_list = [get_result[0]] + shuffle_kernel + [tail]
    return '/'.join(new_url_list)


def segment_attack(domain: str):
    char_list = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    choice_chr = random.sample(char_list, k=50)
    return domain + "#" + "".join(choice_chr)


def port_attack(domain: str):
    file_index = re.findall(domain, "?")
    if file_index:
        print(file_index)
