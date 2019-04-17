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