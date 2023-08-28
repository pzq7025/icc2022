# -*- coding:utf-8 -*-
import random
import re
from urllib.parse import urlparse, urlunparse, urlencode


def get_url_part(url: str):
    url = url.replace("[", "").replace("]", "")
    pre = "http://"
    result = urlparse(pre + url)
    return result


def get_url_part_special(url: str):
    url = url.replace("[", "").replace("]", "")
    result = urlparse(url)
    return result


def get_url_good_part(url: str):
    result = urlparse(url)
    return result


def scheme():
    return 0


with open("./adversarialSampleAnalyse/file_extension.txt", "r") as f:
    file_extension = [i.strip().lower() for i in f.readlines()]

with open("./adversarialSampleAnalyse/url_label.txt", "r") as f:
    url_label = [i.strip().lower() for i in f.readlines()]

with open("./adversarialSampleAnalyse/frequency_word.txt", "r") as f:
    url_words = [i.strip().lower() for i in f.readlines()]

char_list = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]


def params_attack(url: str):
    params = {
        "pid": "".join(random.choices(char_list, k=10)),
        "user": "".join(random.choices(char_list, k=5)),
        "ps": "".join(random.choices(char_list, k=8)),
        "token": "".join(random.choices(char_list, k=12)),
    }
    result = get_url_part(url)
    # print(result)
    new_path = result.path + '/'.join([''.join(random.choices(char_list, k=random.randint(1, 7))) for _ in range(3)]) + '/'
    new_netloc = result.netloc + ":" + str(random.randint(0, 65535))
    # print(new_path)
    other_list = ['-', "_"]
    file_ex = ''.join(random.choices(char_list, k=random.randint(1, 7))) + "_" + ''.join(random.choices(char_list, k=random.randint(1, 4)))
    scheme_list = ["http", "https", "ftp", "gopher", 'mailto', "news", "nntp", "telnet", "wais", "file", "prospero"]
    # 六个
    # new = result._replace(scheme=random.choice(["http", "https", "ftp", "scp"]), path=new_path, params=file_ex + "." + random.choice(file_extension), netloc=new_netloc, query=urlencode(params), fragment="".join([chr(random.randint(35, 122)) for _ in range(10)]))
    # 五个
    new = result._replace(scheme=random.choice(scheme_list), path=new_path, params="filename=" + file_ex + "." + random.choice(file_extension), netloc=new_netloc, query=urlencode(params), fragment="".join([chr(random.randint(35, 122)) for _ in range(10)]))
    # 四个
    # new = result._replace(scheme=random.choice(["http", "https", "ftp", "scp"]), path=new_path, params=file_ex + "." + random.choice(file_extension), netloc=new_netloc, query=urlencode(params), fragment="".join([chr(random.randint(35, 122)) for _ in range(10)]))
    # 三个
    # new = result._replace(scheme=random.choice(["http", "https", "ftp", "scp"]), path=new_path, params=file_ex + "." + random.choice(file_extension), netloc=new_netloc, query=urlencode(params), fragment="".join([chr(random.randint(35, 122)) for _ in range(10)]))
    # 两个
    # new = result._replace(scheme=random.choice(["http", "https", "ftp", "scp"]), path=new_path, params=file_ex + "." + random.choice(file_extension), netloc=new_netloc, query=urlencode(params), fragment="".join([chr(random.randint(35, 122)) for _ in range(10)]))
    # 一个
    # new = result._replace(scheme=random.choice(["http", "https", "ftp", "scp"]), path=new_path, params=file_ex + "." + random.choice(file_extension), netloc=new_netloc, query=urlencode(params), fragment="".join([chr(random.randint(35, 122)) for _ in range(10)]))
    # print(new)
    # print(urlunparse(new))
    return urlunparse(new)


# params_attack("www.capsoftnm.com/main/scap/")
def components_netloc_attack(url: str):
    result = get_url_part(url)
    new_netloc = re.sub(r"[\d+|~|-]", "", result.netloc)
    new_netloc = new_netloc + ":" + str(random.randint(0, 65535))
    new = result._replace(scheme="", netloc=new_netloc)
    return urlunparse(new)[2:]


def components_path_second_attack(url: str):
    result = get_url_part(url)
    new_char = char_list + ["<", ">", "-", "_"]
    path_1 = "".join(random.choices(new_char, k=random.randint(0, 20)))
    new = result._replace(scheme="", query="", fragment="", path=path_1)
    return urlunparse(new)[2:]


def components_path_attack(url: str):
    result = get_url_part(url)
    if not result.path:
        new_char = char_list + ["%", "$", "-", "_", ".", "@", "\\", "?", "~", "=", ":", ".", "+", "|"]
        path_1 = "/".join([''.join(random.choices(new_char, k=random.randint(5, 20))) for _ in range(0, 5)])
    # file_ex = ''.join(random.choices(char_list, k=random.randint(1, 7))) + "_" + ''.join(random.choices(char_list, k=random.randint(1, 4)))
    # params = "filename=" + file_ex + "." + random.choice(file_extension)
        new = result._replace(scheme="", path=path_1)
        # new = result._replace(params=params)
        return urlunparse(new)[2:]
    return url


def components_params_second_attack(url: str):
    result = get_url_part(url)
    new_char_list = char_list + ["%", "$", "-", "_"] + ["|", "\\", "^", "~"] + ["/", "?", ":", "@", "&", "="] + ["<", ">", "#", "%"]
    num_params = 10
    per = random.choices(url_words, k=num_params)
    end = [''.join(random.choices(char_list, k=random.randint(3, 8))) for _ in range(num_params)]
    params_1 = [i + '=' + k for i, k in zip(per, end)]
    new_par = ';'.join(params_1)
    new = result._replace(scheme="", params=new_par)
    # .replace(";", "")
    return urlunparse(new)[2:]


def components_params_attack(url: str):
    result = get_url_part(url)
    if not result.params:
        num_params = 4
        new_char_list = char_list + ["%", "$", "-", "_", ".", "+"] + ["!", "*", "'", "(", ")", ","] + ["{", "}", "|", "\\", "^", "~", "[", "]", "`"] + [";", "/", "?", ":", "@", "&", "="] + ["<", ">", "#", "%", "<", "\""]
        pre_part = [''.join(random.choices(new_char_list, k=random.randint(3, 5))) for _ in range(num_params)]
        back_part = [''.join(random.choices(new_char_list, k=random.randint(3, 5))) for _ in range(num_params)]
        new_param = ';'.join([pre + "=" + back for pre, back in zip(pre_part, back_part)])
        file_ex = ''.join(random.choices(char_list, k=random.randint(1, 7))) + random.choice(["_", "-"]) + ''.join(random.choices(char_list, k=random.randint(1, 4)))
        file_param = "filename=" + file_ex + "." + random.choice(file_extension)
        params = file_param + ";" + new_param
        # new_path = result.path + '/'.join([''.join(random.choices(char_list, k=random.randint(1, 7))) for _ in range(3)]) + '/'
        new = result._replace(scheme="", params=params)
        return urlunparse(new)[2:].replace(";", "")
    return url.replace(";", "")


def components_query_second_attack(url: str):
    result = get_url_part(url)
    # new_char_list = char_list + ["%", "$", "-", "_", ".", "+"] + ["!", "*", "'", "(", ")", ","] + ["{", "}", "|", "\\", "^", "~", "[", "]", "`"] + [";", "/", "?", ":", "@", "&", "="] + ["<", ">", "#", "%", "<", "\""]
    #
    new_char_list = char_list + ["%", "$", "-", "_", ".", "+", "#", "?", "<", ">", "/"]
    # params = {
    #     ''.join(random.choices(char_list, k=random.randint(2, 5))): "".join(random.choices(char_list, k=random.randint(2, 10))),
    #     ''.join(random.choices(char_list, k=random.randint(2, 5))): "".join(random.choices(char_list, k=random.randint(2, 10))),
    #     ''.join(random.choices(char_list, k=random.randint(2, 5))): "".join(random.choices(char_list, k=random.randint(2, 10))),
    #     "token": "".join(random.choices(char_list, k=12)),
    # }
    params = {
        random.choice(url_words): "".join(random.choices(char_list, k=random.randint(2, 10))),
        random.choice(url_words): "".join(random.choices(char_list, k=random.randint(2, 10))),
        random.choice(url_words): "".join(random.choices(char_list, k=random.randint(2, 10))),
        random.choice(url_words): "".join(random.choices(char_list, k=12)),
    }
    new_query = urlencode(params)
    new = result._replace(scheme="", query=new_query, params="")
    return urlunparse(new)[2:]


def components_query_attack(url: str):
    result = get_url_part(url)
    char_list.extend(["%", "$", "-", "_", ".", "+"])
    char_list.extend(["!", "*", "'", "(", ")", ","])
    char_list.extend(["{", "}", "|", "\\", "^", "~", "[", "]", "`"])
    char_list.extend([";", "/", "?", ":", "@", "&", "="])
    char_list.extend(["<", ">", "#", "%", "<", "\""])
    params = {
        "pid": "".join(random.choices(char_list, k=10)),
        "user": "".join(random.choices(char_list, k=5)),
        "ps": "".join(random.choices(char_list, k=8)),
        "token": "".join(random.choices(char_list, k=12)),
    }
    if not result.query:
        new_query = urlencode(params)
    # else:
    #     #     new_query = ""
        new = result._replace(scheme="", query=new_query)
        return urlunparse(new)[2:]
    return url


def components_fragment_second_attack(url: str):
    result = get_url_part(url)
    new_fragment = "".join([chr(random.randint(35, 122)) for _ in range(10)])
    new_fragment = '.>#'.join(random.choices(url_label, k=random.randint(10, 20)))
    # new = result._replace(scheme="", fragment=new_fragment)
    new = result._replace(scheme="", fragment=new_fragment)
    return urlunparse(new)[2:]


def components_fragment_attack(url: str):
    result = get_url_part(url)
    if not result.fragment:
        new_char_list = char_list + ["%", "$", "-", "_", ".", "+"] + ["!", "*", "'", "(", ")", ","] + ["{", "}", "|", "\\", "^", "~", "[", "]", "`"] + [";", "/", "?", ":", "@", "&", "="] + ["<", ">", "#", "%", "<", "\""]
        new_fragment = "".join(random.choices(new_char_list, k=10))
        new = result._replace(scheme="", fragment=new_fragment)
        return urlunparse(new)[2:]
    return url


def components_test_attack(url: str):
    # result = get_url_part(url)
    # new = result._replace(path="", params="", query="", fragment="")
    # return urlunparse(new)
    result = get_url_part(url)
    # new_netloc = re.sub(r"[\d+|~|-]", "", result.netloc)
    # netloc = result.netloc.replace(r"\w", ""), netloc=new_netloc
    params_1 = [''.join(random.choices(char_list, k=random.randint(0, 5))) for _ in range(3)]
    new_par = '.'.join(params_1)
    new_char = char_list + ["<", ">", "-", "_"]
    path_1 = "".join(random.choices(new_char, k=random.randint(0, 20)))
    new = result._replace(scheme="", params=new_par, query="", fragment="", path=path_1)
    # if result.params:
    #     file_ex = ''.join(random.choices(char_list, k=random.randint(1, 7))) + "_" + ''.join(random.choices(char_list, k=random.randint(1, 4)))
    #     params = "filename=" + file_ex + "." + random.choice(file_extension)
    #     new = result._replace(params=params)
    #     return urlunparse(new)
    return urlunparse(new)[2:]


def components_scheme_attack(url: str):
    result = get_url_part(url)
    scheme_list = ["http", "https", "ftp", "gopher", 'mailto', "news", "nntp", "telnet", "wais", "file", "prospero"]
    new = result._replace(scheme=random.choice(scheme_list))
    return urlunparse(new)


total = 1


def total_params_attack(url: str):
    result = get_url_part(url)
    new_fragment = '.>#'.join(random.choices(url_label, k=random.randint(10, 20)))
    new_netloc1 = re.sub(r"[\d+|~|-]", "", result.netloc)
    new_netloc = new_netloc1 + ":" + str(random.randint(0, 65535))
    scheme_list = ["http", "https", "ftp", "gopher", 'mailto', "news", "nntp", "telnet", "wais", "file", "prospero"]
    char_list.extend(["%", "$", "-", "_", ".", "+"])
    char_list.extend(["!", "*", "'", "(", ")", ","])
    char_list.extend(["{", "}", "|", "\\", "^", "~", "[", "]", "`"])
    char_list.extend([";", "/", "?", ":", "@", "&", "="])
    char_list.extend(["<", ">", "#", "%", "<", "\""])
    params = {
        "pid": "".join(random.choices(char_list, k=10)),
        "user": "".join(random.choices(char_list, k=5)),
        "ps": "".join(random.choices(char_list, k=8)),
        "token": "".join(random.choices(char_list, k=12)),
    }
    if not result.query:
        new_query = urlencode(params)
        result = result._replace(query=new_query)
    if not result.path:
        new_char = char_list + ["%", "$", "-", "_", ".", "@", "\\", "?", "~", "=", ":", ".", "+", "|"]
        path_1 = "/".join([''.join(random.choices(new_char, k=random.randint(5, 20))) for _ in range(0, 5)])
    # file_ex = ''.join(random.choices(char_list, k=random.randint(1, 7))) + "_" + ''.join(random.choices(char_list, k=random.randint(1, 4)))
    # params = "filename=" + file_ex + "." + random.choice(file_extension)
        result = result._replace(path=path_1)
        # new = result._replace(params=params)
    if not result.params:
        num_params = 4
        new_char_list = char_list + ["%", "$", "-", "_", ".", "+"] + ["!", "*", "'", "(", ")", ","] + ["{", "}", "|", "\\", "^", "~", "[", "]", "`"] + [";", "/", "?", ":", "@", "&", "="] + ["<", ">", "#", "%", "<", "\""]
        pre_part = [''.join(random.choices(new_char_list, k=random.randint(3, 5))) for _ in range(num_params)]
        back_part = [''.join(random.choices(new_char_list, k=random.randint(3, 5))) for _ in range(num_params)]
        new_param = ';'.join([pre + "=" + back for pre, back in zip(pre_part, back_part)])
        file_ex = ''.join(random.choices(char_list, k=random.randint(1, 7))) + random.choice(["_", "-"]) + ''.join(random.choices(char_list, k=random.randint(1, 4)))
        file_param = "filename=" + file_ex + "." + random.choice(file_extension)
        params = file_param + ";" + new_param
        # new_path = result.path + '/'.join([''.join(random.choices(char_list, k=random.randint(1, 7))) for _ in range(3)]) + '/'
        result = result._replace(params=params)
    new = result._replace(scheme=random.choice(scheme_list), fragment=new_fragment, netloc=new_netloc)
    return urlunparse(new)

