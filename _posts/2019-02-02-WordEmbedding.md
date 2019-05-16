---
layout: post
title:  "Word Embedding"
categories: DeepLearning
tags: NLP
---

* content
{:toc}

## Skip-Gram

Skip-Gram是**词嵌入**(word embedding)的经典模型，其思想是在向量空间中，以中心词为输入，最大化上下文单词的相似性。

在原始文本空间中，以指定长度的词序列为窗口，相应的概率如下图所示：

![](/img/2019-04-17_15-33-51.bmp)

对整段文字而言，其在参数$\theta$下的似然函数为：

$$
L(\theta)=\prod\limits_{t=1}^{T}\prod\limits_{-m{\le}j{\le}m\atop{j{\ne}0}}P(w_{t+j}|w_{t};\theta)
$$

对数似然函数为：

$$
\ln{L(\theta)}=\sum\limits_{t=1}^{T}\sum\limits_{-m{\le}j{\le}m\atop{j{\ne}0}}P(w_{t+j}|w_{t};\theta)
$$

那么需要最小化的目标函数为：

$$
J(\theta)=-\frac{1}{T}\sum\limits_{t=1}^{T}\sum\limits_{-m{\le}j{\le}m\atop{j{\ne}0}}P(w_{t+j}|w_{t};\theta)
$$

其中$T$为文本总长度，$m$为窗口尺寸，$w_{t}$为中心词，$w_{t+j}$为上下文。

一个最简单的wordEmbedding示例见[这里](https://github.com/Daya-Jin/DL_for_learner/blob/master/NLP/WordEmbedding.ipynb)。

## word2vec

word2vec改变了损失函数，引入负采样技术，将多分类softmax损失转成了计算二分类log损失：

$$
J(\theta)=-\frac{1}{T}\sum\limits_{t=1}^{T}\log{P_{\theta}(D=1|pair_{pos})}+\log{P_{\theta}(D=0|pair_{neg})}
$$

其中$D$代表词组对的相邻性，如果两单词具有上下文关系，即一者是另一者的中心词，则$D=1$，否则$D=0$。

一个正儿八经的word2vec示例[见此](https://github.com/Daya-Jin/DL_for_learner/blob/master/NLP/word2vec.ipynb)。