---
layout: post
title:  "Word Embedding"
categories: DeepLearning
tags: NLP
---

* content
{:toc}

# 概述

对于文本单词的表示，最经典的方法是使用**词袋**(bag of word)模型，即对于每一个单词，都为其分配一个唯一的标识ID。如一个句子```I am a king```，将所有单词排序后分配一个唯一标识，那么对应关系有：

|Word|ID|
|:-:|:-:|
|a|0|
|am|1|
|I|2|
|king|3|

如果要使用机器学习或深度学习的技术去分析文本数据时，那么自然而然会将这些单词转换成one-hot向量输送到模型中去。如果以单词为单位，上述句子所构成的训练集可以表示成：

$$
\left[
 \begin{matrix}
   0 & 0 & 1 & 0 \\
   0 & 1 & 0 & 0 \\
   1 & 0 & 0 & 0 \\
   0 & 0 & 0 & 1 \\
  \end{matrix}
\right]
$$

文本数据中出现的独特单词构成的集合叫做**词汇**(vocabulary)或**词典**(dictionary)，如上述句子的词汇大小就为$4$。当使用one-hot方式来表示单词时，易得每个单词的向量长度为词汇大小```voc_size```。这种方法的缺点显而易见，在语料库过大时，词汇大小是无法预料的，这就会导致词向量的长度过大并且是极度稀疏的。

引入隐语义或隐特征的概念，假设现在有六个单词：man, woman, king, queen, apple, orange。那么可以根据这些的单词所具有的一些属性写出一些特征：

|Word|Gender|Royal|Age|Food|
|:-:|:-:|:-:|:-:|:-:|
|man|-1|0.01|0.03|0.09|
|woman|1|0.02|0.02|0.01|
|king|-0.95|0.93|0.7|0.02|
|queen|0.97|0.95|0.69|0.01|
|apple|0|-0.01|0.03|0.95|
|orange|0.01|0|-0.02|0.97|

藉由隐特征的表示方式，词向量之间很好地保留了其语义信息，并且这种表示方式没有稀疏的缺点。WordEmbedding的核心思想就是如何使用机器学习或深度学习技术去学习得到单词的这种表示。

在实际实现(TensorFlow)中，词的嵌入表示是使用一个矩阵来存储的。假设词汇大小为$\vert{V}\vert$，嵌入维度为$\vert{E}\vert$，那么嵌入表示的矩阵形状就为$(\vert{V}\vert, \vert{E}\vert)$，矩阵的形状同神经网络中的一样。那么查找过程是这样的，假设上述例子中的```king```的onehot表示为：

$$
x_{king}=
\left[
 \begin{matrix}
   0 & 0 & 1 & 0 & 0 & 0 \\
  \end{matrix}
\right]
$$

嵌入矩阵为：

$$
emb\_lookup=
\left[
 \begin{matrix}
   X & X & X & X \\
   X & X & X & X \\
   -0.95 & 0.93 & 0.7 & 0.02 \\
   X & X & X & X \\
   X & X & X & X \\
   X & X & X & X \\
  \end{matrix}
\right]
$$

```king```的嵌入表示可以通过矩阵相乘得到：

$$
e_{king}=x_{king}\times{emb\_lookup}=
\left[
 \begin{matrix}
   -0.95 & 0.93 & 0.7 & 0.02 \\
  \end{matrix}
\right]
$$

不难发现，当使用一个词的onehot向量与嵌入矩阵相乘时，其实就相当于取出嵌入矩阵的某一行。因为onehot向量的特殊性，其与任何矩阵相乘起到的是一个查询某行的作用，因此在实现中嵌入矩阵通常叫做```emb_lookup```。

## Skip-Gram

Skip-Gram是**词嵌入**(word embedding)的经典模型，其思想是在向量空间中，以中心词为输入，最大化上下文单词的相似性。

在原始文本空间中，以指定长度的词序列为窗口，相应的概率如下图所示：

![](/img/2019-04-17_15-33-51.bmp)

对整段文字而言，其在参数$\theta$下的似然函数为：

$$
L(\theta)=\prod\limits_{t=1}^{T}\prod\limits_{-m{\le}j{\le}m\atop{j{\ne}0}}P(w_{t+j}\vert{w_{t}};\theta)
$$

对数似然函数为：

$$
\ln{L(\theta)}=\sum\limits_{t=1}^{T}\sum\limits_{-m{\le}j{\le}m\atop{j{\ne}0}}P(w_{t+j}\vert{w_{t}};\theta)
$$

那么需要最小化的目标函数为：

$$
J(\theta)=-\frac{1}{T}\sum\limits_{t=1}^{T}\sum\limits_{-m{\le}j{\le}m\atop{j{\ne}0}}P(w_{t+j}\vert{w_{t}};\theta)
$$

其中$T$为文本总长度，$m$为窗口尺寸，$w_{t}$为中心词，$w_{t+j}$为上下文。

一个最简单的wordEmbedding示例见[这里](https://github.com/Daya-Jin/DL_for_learner/blob/master/NLP/WordEmbedding.ipynb)。

朴素Skip-Gram的问题在哪里？问题在于$P(w_{t+j}\vert{w_{t}};\theta)$的计算。根据一个词去预测另一个的概率，实际上就是一个多分类问题，多分类问题的输出一般是使用$softmax$函数来计算的，其输出是一个长度等于类别数的概率向量。对于单词预测，类别数就是词汇表的大小，假设词汇大小为$V$，单词$w_{i}$的嵌入表示为$z_{i}$，那么给定一个中心词$w_{1}$，预测某一个上下文单词$w_{2}$的出现概率为：

$$
P(w_{2}\vert{w_{1}};\theta)=\frac{e^{\theta_{2}^{T}z_{2}}}{\sum_{i=1}^{V}e^{\theta_{i}^{T}z_{i}}}
$$

注意这只是$softmax$层输出向量中的一个标量而已，完整的$softmax$层输出为：

$$
\left[
 \begin{matrix}
   P(w_{1}\vert{w_{1}};\theta) & P(w_{2}\vert{w_{1}};\theta) & \cdots & P(w_{V}\vert{w_{1}};\theta) \\
  \end{matrix}
\right]
$$

不难发现$softmax$层的计算量非常巨大，因为词汇大小$V$是巨大的。

## word2vec

为了优化Skip-Gram在计算softmax时的计算负担，word2vec改变了预测时的思路。Skip-Gram模型的思路是使用一个中心词去预测上下文词，word2vec将其转成了预测两个单词是否具有上下文关系，即把多分类问题转成了二分类问题。解决extreme multiclass问题所用到的技术叫**负采样**(negative sampling)。

假设文本数据中有这么一段：

> ... a glass of orange juice ...

假设以```orange```为中心词，窗口为$1$，那么能得到一个正样本

> ```[(orange, juice), 1]```

这里暂时不考虑```[orange, of]```。假设负采样率$k=3$，那么在词汇中取出$3$个与```orange```无上下文关系的单词组成三个负样本：

> ```[(orange, king), 0]```
> 
> ```[(orange, book), 0]```
> 
> ```[(orange, boy), 0]```

那么现在现在只需要计算$k+1$个概率：

$$
\left[
 \begin{matrix}
   P(juice\vert{orange};\theta) & P(king\vert{orange};\theta) & P(book\vert{orange};\theta) & P(boy\vert{orange};\theta) \\
  \end{matrix}
\right]
$$

在该轮只需要把这四个输出当做是四个二分类器来更新参数即可。损失函数可以写成：

$$
J(\theta)=-\frac{1}{N}\sum\limits_{n=1}^{N}[\log{P_{\theta}(D=1\vert{pair_{pos}})}+\log{P_{\theta}(D=0\vert{pair_{neg}})}]
$$

其中$D$代表词组对的关系，如果两单词具有上下文关系则$D=1$，反之$D=0$；$pair_{pos}$代表正样本单词对，$pair_{neg}$代表负样本单词对；$N$表示样本数。

那么如何选取单词区构成负样本？原论文中给出的建议是，在生成负样本时，单词$w_{i}$被选中的概率为：

$$
P(w_{i})=\frac{f(w_{i})^{3/4}}{\sum\limits_{j=1}^{\vert{V}\vert}f(w_{j})^{3/4}}
$$

其中$f(w_{i})$表示单词$w_{i}$在整个文本数据中出现的频数；$\vert{V}\vert$表示词汇大小。

一个正儿八经的word2vec示例[见此](https://github.com/Daya-Jin/DL_for_learner/blob/master/NLP/word2vec.ipynb)。
