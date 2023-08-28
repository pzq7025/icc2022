# -*- coding:utf-8 -*-


def record_important_layer(dict_reflect, choose, str_length):
    """
    记录字符的重要程度
    :dict_reflect：总的字符统计表
    :choose  被替换的字符表，选择需要替换的字符表
    ：str_length：总的url长度
    记录所有字符的重要程度，通过频数除以总数
    选择的字符在这个频数中的重要程度，以此衡量替换的可靠性
    :return:
    """
    _ = list(map(lambda x: {x[0]: round(x[1] / str_length, 4)}, dict_reflect.items()))
    query_table = {}
    for i in _:
        query_table.update(i)
    return [query_table.get(i) for i in choose]


def store_file(data: list, file_name: str):
    """
    存储扰动的分布数据
    :param data:
    :param file_name:
    :return:
    """
    with open(file_name, 'w') as f:
        f.writelines([str(i) + '\n' for i in data])
    print("存贮成功")
