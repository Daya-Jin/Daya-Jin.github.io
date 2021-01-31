---
layout: post
title:  "FM Families"
categories: MachineLearning
tags: FM
---

* content
{:toc}

## Factorization Machine

在线性回归模型的博客中，提到了FM模型的表达式为：

$$
\hat{y}=\theta_{0}+\sum_{i=1}^{m}\theta_{i}x_{i}+\sum_{i=1}^{m}\sum_{j=1}^{m}<v_{i},v_{j}>x_{i}x_{j}
$$

其中$v_{i}$为矩阵$V_{m\times{k}}$的第$i$行。上式可以变换成：

$$
\hat{y}=\theta_{0}+\sum_{i=1}^{m}\theta_{i}x_{i}+\frac{1}{2}\sum_{j=1}^{k}((\sum\limits_{i=1}^{m}v_{ij}x_{i})^{2}-\sum\limits_{i=1}^{m}v_{ij}^{2}x_{i}^{2})
$$
