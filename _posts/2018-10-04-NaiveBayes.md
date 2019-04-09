---
layout: post
title:  "Naive Bayes"
categories: MachineLearning
tags: bayes
---

* content
{:toc}

# 模型概述

首先回顾一下贝叶斯公式：

$$
P(A|B)=\frac{P(AB)}{P(B)}=\frac{P(B|A)P(A)}{P(B)}
$$

以二分类为例，上述公式以机器学习任务的形式来写的话就成为了：

$$
P(y=0|x)=\frac{P(x|y=0)P(y=0)}{P(x)} \\
P(y=1|x)=\frac{P(x|y=1)P(y=1)}{P(x)} \\
$$

其中$x$为待预测样本。对于需要算的几个概率，一个一个来看。

$P(x|y=0)$，注意到样本$x$是一个同时有多个值的向量，$x=
\left[
\begin{matrix}
 x_{0} & x_{1} & \cdots & x_{n}
\end{matrix}
\right]$，在数据集中很可能没有跟待预测样本$x^{(i)}$完全相同的样本，那么就没法直接计算$P(x|y=0)$。注意到在**各特征相互独立**的前提下，有：

$$
P(x|y=0)=P(x^{D}_{0}=x_{0}|y=0)P(x^{D}_{1}=x_{1}|y=0)...P(x^{D}_{n}=x_{n}|y=0)
$$

$P(y=0)$，这个好办，直接计算样本中负样本出现的频率，相对应地，$P(y=1)$即样本中正样本出现的频率。

$P(x)$这个概率同样不好直接计算，根据全概率公式，有：

$$
P(x)=P(y=0)P(x|y=0)+P(y=1)P(x|y=1)
$$

在各特征独立的条件下，上式可以写成：

$$
\begin{aligned}
P(x)&=P(y=0)P(x^{D}_{0}=x_{0}|y=0)...P(x^{D}_{n}=x_{n}|y=0) \\
&+P(y=1)P(x^{D}_{0}=x_{0}|y=1)...P(x^{D}_{n}=x_{n}|y=1) \\
\end{aligned}
$$

容易看出，对同一个数据集而言，$P(x)$是不变的，所以只需要关注分子即可。

所以上述问题在多分类的情况下可以用以下公式来表达：

$$
\hat{y}=arg\ \max\limits_{c_{j}}\ P(Y=c_{k})\prod\limits_{i=0}^{n}P(x_{i}^{D}=x_{i}|Y=c_{j})
$$

其中$x$为待预测样本，$\hat{y}$为模型输出，$x_{i}^{D}$为数据集中的样本的第$i$个特征，$Y$为数据集标签，$c_{j}$为第$j$个类别。同时注意到上面做了两次假设：**各特征之间相互独立**，这是朴素贝叶斯最重要的一个前提条件。

## 连续属性

对于数据集中的连续属性
$x_{i}^{D}$，怎么计算$P(x_{i}^{D}=x_{i}|Y=c_{k})$？可假设该连续特征在某一类别下$c_{k}$服从某一分布，如高斯分布$P(x_{i}|Y=c_{k})=\frac{1}{\sqrt{2\pi}\sigma_{c_{k},i}}exp(-\frac{(x_{i}-\mu_{c_{k},i})^{2}}{2\sigma_{c_{k},i}^{2}})$。

## 一点改进

在原始的问题公式中，如果累乘项中的某一项为零，那么就会影响最终结果从而始终得到零概率输出，如有一项
$P(x_{i}^{D}=x_{i}|Y=c_{j})=0$，则不管该样本的其他属性如何，模型对该样本属于各个类别的预测概率均为0，这说明模型没有很好的泛化能力。有两种改进方法：

1. 将累乘取对数转换成累加

2. 拉普拉斯修正

   原概率计算公式为
   $P(Y=c_{k})=\frac{|D_{c_{k}}|}{|D|}$，
   $P(x_{i}^{D}=x_{i}|Y=c_{k})=\frac{|D_{c_{k},x_{i}}|}{|D_{c_{k}}|}$，经拉普拉斯修正后的概率计算公式为$\hat{P}(Y=c_{k})=\frac{|D_{c_{k}}|+1}{|D|+N}$，$\hat{P}(x_{i}^{D}=x_{i}|Y=c_{k})=\frac{|D_{c_{k},x_{i}}|+1}{|D_{c_{k}}|+N_{i}}$，其中$N$为数据集的类别数，$N_{i}$为第$i$个特征的可能取值数。

[实现指导](https://github.com/Daya-Jin/ML_for_learner/blob/master/naive_bayes/Gaussian%20Naive%20Bayes.ipynb)

[完整代码](https://github.com/Daya-Jin/ML_for_learner/blob/master/naive_bayes/GaussianNB.py)
