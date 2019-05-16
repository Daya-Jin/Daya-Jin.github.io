---
layout: post
title:  "Text Classification"
categories: NLP
tags: NLP DeepLearning
---

* content
{:toc}

# 概述

原始文本文件在项目目录下的```./dataset/news_CN/```下，每一行的格式为```{label}\t{text}```，如：

```时政\t台风莫拉克重创台湾南部 15人死亡65人失踪```

## 预处理

### 分词

对于文本任务，最基本的预处理就是分词，这里使用```jieba```开源分词库来完成。

```python
def gen_seg_file(file_in, file_out):
    '''
    生成分词后的文件
    :param file_in: 原始未分词的文件
    :param file_out: 输出文件，词语使用' '分隔
    :return:
    '''
    with open(file_in, 'r', encoding='utf-8') as fd:
        text = fd.readlines()
    with open(file_out, 'w', encoding='utf-8') as fd:
        for line in text:
            label, data = line.strip().split('\t')
            words = jieba.cut(data)
            words_trans = ''

            # 去除切分出来的空白词
            for word in words:
                word = word.strip()
                if word != '':
                    words_trans += word + ' '

            out_line = '{}\t{}\n'.format(label, words_trans.strip())
            fd.write(out_line)
```

### 词典

分词之后，需要对数据做格式化处理。那么最简单的格式化就是对每一个单词做整形编码，每一个单词对应着唯一的一个数字。对于label而言同样需要做格式化。

为了实现整形编码，需要构建一个词典，即单词与数字的映射表，还有类别与数字的映射表。同时注意到，一个包含所有可能单词的词典是巨大的，实际中不可能接受这样大的存储开销，所以实际的词典只会记录一部分词语，这里选择按频数来选择记录哪些词语。除此之外，词典中还必须能够对未知词语编码，这里对未知词语统一编码成$0$。

```python
def gen_vocab(file_in, file_out):
    '''
    生成词典文件，每行格式为'idx word word_cnt'
    :param file_in:
    :param file_out:
    '''
    with open(file_in, 'r', encoding='utf-8') as fd:
        text = fd.readlines()

    word_dict = dict()
    for line in text:
        _, data = line.strip().split('\t')
        for word in data.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
    word_dict = sorted(word_dict.items(), key=lambda x: x[1],  # 以频数排序
                       reverse=True)

    with open(file_out, 'w', encoding='utf-8') as fd:
        fd.write('0\t<UNK>\t99999\n')
        for idx, item in enumerate(word_dict):
            fd.write('{}\t{}\t{}\n'.format(idx + 1, item[0], item[1]))
```

类别词典的构建就比较简单了，直接做一一映射即可：

```python
def gen_cat(file_in, file_out):
    '''
    生成类别编码文件
    :param file_in:
    :param file_out:
    :return:
    '''
    with open(file_in, 'r', encoding='utf-8') as fd:
        text = fd.readlines()

    label_dict = dict()
    for line in text:
        label, _ = line.strip().split('\t')
        label_dict.setdefault(label, 0)
        label_dict[label] += 1
    label_dict = sorted(label_dict.items(), key=lambda x: x[1],
                        reverse=True)

    with open(file_out, 'w', encoding='utf-8') as fd:
        for idx, item in enumerate(label_dict):
            fd.write('{}\t{}\t{}\n'.format(idx, item[0], item[1]))
```

至此，对于原始文件的预处理就结束了。

## 编码

对正文跟label，分别封装两个编(解)码器。

### Text Encoder

对于文本编码器，需要实现编码与解码，同时还要满足单词与句子级别的功能。编码与解码分别通过两个字典实现：

```python
self._word2id = dict()
self._id2word = dict()
```

然后对外暴露的核心API有四个：

```python
def word2id(self, word: str):
    '''
    单次级别的编码
    :param word:
    :return:
    '''
    return self._word2id.get(word, self._unk)

def id2word(self, idx: int):
    '''
    单次级别的解码
    :param idx:
    :return:
    '''
    return self._id2word.get(idx, '<UNK>')

def s2id(self, s: str):
    '''
    句子级别编码
    :param s:
    :return:
    '''
    return [self.word2id(word) for word in s.split(' ')]

def id2s(self, idxs) -> str:
    '''
    句子级别解码
    :param idxs:
    :return:
    '''
    return ' '.join([self.id2word(idx) for idx in idxs])
```

### Label Encoder

类似地，类别编码器的实现也是依靠字典：

```python
self._cat2id = dict()
```

暴露的核心API为编码器：

```python
def cat2id(self, cat):
    if cat not in self._cat2id:
        raise Exception('{} is not in cat'.format(cat))
    else:
        return self._cat2id[cat]
```

## 数据类

与之前实现的一些CNN实例一样，为了便于数据的管理，创建一个```Data```类，数据会被读取到该类中，同时这个类也负责产生batch，其核心API为```next_batch()```。

注意在处理时序数据时，feed到网络中的每一条数据维度(时间维度与特征维度)应该相同。所以对于超出长度的数据，要做截断；而对于长度不足的数据，要做填充。

```python
label, content = line.strip().split('\t')
x = self._vocal.s2id(content)
y = self._cat_dict.cat2id(label)

x = x[:self._t_size]
n_pad = self._t_size - len(x)  # 需要填充的位数
x = x + [self._vocal.unk for _ in range(n_pad)]
```

上述代码中，当```n_pad<=0```时，最后一行的列表生成式不会生效。

## 模型设计

文本分类问题，实际属于RNN中的many to one问题。即RNN部分的输入$rnn\_inputs$具有多个时间状态，RNN部分的输出$rnn\_outputs$只取最后一个时间状态的输出。

同时对于文本的处理，embedding是不可绕开的操作。那么设计一个简单的LSTM网络，首先是对输入$X$做embedding，得到$X_emb$，然后将$X_emb$输送到LSTM网络中，后接FC层，然后得出分类结果。模型结构如下图所示：

![](/img/TextClf.svg)

确定网络结构之后，只需要注意每一层数据流的维度即可。

## 模型搭建

首先是```placeholder```，作为文本输入的$X$拥有时间维度，而预测的目标变量是一个标量。

```python
X = tf.placeholder(tf.int32, [None, params.t_size])
Y = tf.placeholder(tf.int64, [None])
```

而嵌入层的输入维度是onehot向量的维度，输出维度是嵌入维度。对文本数据而言，onehot向量的维度等于词典的大小。

```python
emb_lookup = tf.get_variable('embedding', [vocal_size, params.emb_size],
                                dtype=tf.float32)
emb = tf.nn.embedding_lookup(emb_lookup, X)    # (batch_size,t_size,emb_size)
```

然后是LSTM层：

```python
lstm_layers = list()
for i in range(params.lstm_layers):
    layer = tf.nn.rnn_cell.LSTMCell(params.lstm_size[i])
    lstm_layers.append(layer)

lstm_layers = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
```

RNN的多对一问题，只取出RNN网络最后一层的最后一个时间状态下的输出：

```python
lstm_outputs, _ = tf.nn.dynamic_rnn(lstm_layers,
                                    inputs=emb, dtype=tf.float32)
lstm_outputs = lstm_outputs[:, -1, :]
```

后接FC层：

```python
fc = tf.layers.dense(lstm_outputs, params.fc_size, activation=tf.nn.relu)
```

最终输出：

```python
logits = tf.layers.dense(fc, unit_O, activation=None)    # 输出层，无激活
```

多分类任务，使用softmax损失函数：

```python
loss = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=logits)
```

以上即是核心代码，完整代码[见此](https://github.com/Daya-Jin/DL_for_learner/tree/master/NLP/text_clf)。
