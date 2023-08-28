# 总览
2022年ICC论文[Crafting Text Adversarial Examples to Attack the Deep-Learning-based Malicious URL Detection](https://ieeexplore.ieee.org/document/9838536
)
### 一、 文件介绍
| 文件名/文件夹 | 描述 |
| :----: | :----: |
| AdversarialSampleAnalyse | 构建所需要的资源|
| datafiel | 数据集|
| model | 词表和tokenzier |
| module | 使用的模块 |
| module/model_adv | 对抗攻击的方法 |
| module/model_classification |  分类模型 |
| attack_model_fgm.py | 攻击fgm的模型 |
| attack_model_pgd.py | 攻击pgd的模型 |
| attack_model_none.py | 攻击干净的模型 |
| fgm_file.py | fgm的对抗训练 |
| file_none.py | 直接训练源模型 |
| first_attack.py | 针对论文中第一个数据集的攻击 |
| model_attack_method.py | 对抗攻击的方法 |
| pgd_file.py | fgm的对抗训练 |
| second_attack.py | 针对论文中第二个数据集的攻击 |
| url_attack.py | 针对URL组件的攻击 |
| utils.py | 过程中使用的工具 |
### 二、 使用
#### 2.1 环境
安装一下的环境：
```sh
pip install torch tensorflow scikit-learn
```
#### 2.2 干净的模型
运行```first_attack.py```选择```file_none.py```中的pad_tensor不进行任何训练得到干净的模型。
#### 2.3 对抗训练的模型
运行```first_attack.py```选择```fgm_file.py/pgd_file.py```中的pad_tensor得到对抗训练的模型。
#### 2.4 对抗攻击
加载模型，运行```first_attack.py```选择不同的攻击级别进行攻击。
### 三、 注意
参数化做的不够好，需要自己手动修改方式。另外有torch和TensorFlow的混合使用。具体为：使用了TensorFlow进行数据处理，在用torch模型进行训练。
### 四、引用
```text
@INPROCEEDINGS{9838536,
  author={Peng, Zuquan and He, Yuanyuan and Sun, Zhe and Ni, Jianbing and Niu, Ben and Deng, Xianjun},
  booktitle={ICC 2022 - IEEE International Conference on Communications}, 
  title={Crafting Text Adversarial Examples to Attack the Deep-Learning-based Malicious URL Detection}, 
  year={2022},
  volume={},
  number={},
  pages={3118-3123},
  doi={10.1109/ICC45855.2022.9838536}}
```