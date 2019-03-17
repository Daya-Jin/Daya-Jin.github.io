---
layout: post
title:  "Gaussian Mixture Models"
categories: MachineLearning
tags: GMM clustering
---

* content
{:toc}

## 算法概述

**高斯混合模型**(Gaussian Mixture Models)是一种无监督聚类模型。GMM认为不同类别的特征密度函数是不一样的(实际上也不一样)，GMM为每个类别下的特征分布都假设了一个服从高斯分布的概率密度函数：

$$
\begin{aligned}
    P(x|c_{k})&=\frac{1}{\sqrt{2\pi}\sigma_{k}}exp(-\frac{(x-\mu_{k})^{2}}{2\sigma_{k}^2}) \\
    P(x|c_{k})&{\sim}N(\mu_{k},\sigma_{k}) \\
\end{aligned}
$$

而数据中又可能是由多个类混合而成，所以数据中特征的概率密度函数可以使用多个高斯分布的组合来表示：

$$
\begin{aligned}
    P(x)&=\sum\limits_{k=1}^{K}P(c_{k})P(x|c_{k}) \\
        &=\sum\limits_{k=1}^{K}\pi_{k}N(x|\mu_{k},\sigma_{k}) \\
\end{aligned}
$$

其中$\pi_{k}$为类分布概率，也可看做是各高斯分布函数的权重系数，也叫做**混合系数**(mixture coefficient)，其满足$\sum_{k=1}^{K}\pi_{k}=1$。

## Expectation-Maximization

模型的形式有了，给定一组数据$X$，我们需要得到一组参数$\{\mu,\sigma\}$，使得在这组参数下观测数据$$X$$出现的概率最大，即最大似然估计。对于数据中的所有样本，其出现的概率(似然函数)为：

$$
\prod\limits_{i=1}^{N}P(x_{i})=\prod\limits_{i=1}^{N}\sum\limits_{k=1}^{K}\pi_{k}N(x_{i}|\mu_{k},\sigma_{k})
$$

对数似然函数为：

$$
\sum\limits_{i=1}^{N}\ln\{\sum\limits_{k=1}^{K}\pi_{k}N(x_{i}|\mu_{k}\sigma_{k})\}
$$

假设我们现在有了参数$\{\mu,\sigma\}$，需要计算某个样本对应的类簇，由贝叶斯公式有：

$$
\begin{aligned}
    P(c_{k}|x_{i})&=\frac{P(c_{k},x_{i})}{P(x_{i})} \\
    &=\frac{P(x_{i}|c_{k})P(c_{k})}{P(x_{i})} \\
    &=\frac{\pi_{k}N(x_{i}|\mu_{k},\sigma_{k})}{\sum\limits_{k=1}^{K}\pi_{k}N(x_{i}|\mu_{k},\sigma_{k})}
\end{aligned}
$$

可以看出就是一个softmax的形式。同时，有了$P(c_{k}\|x_{i})$之后，又可以计算出某个类别的分布概率与该类别下的统计量：

$$\begin{aligned}
    N_{k}&=\sum\limits_{i=1}^{N}P(c_{k}|x_{i}) \\
    \pi_{k}&=\frac{N_{k}}{N}=\frac{1}{N}\sum\limits_{i=1}^{N}P(c_{k}|x_{i}) \\
    \mu_{k}&=\frac{1}{N_{k}}\sum\limits_{i=1}^{N}P(c_{k}|x_{i})x_{i} \\
    \sigma_{k}&=\sqrt{\frac{1}{N_{k}}\sum\limits_{i=1}^{N}P(c_{k}|x_{i})(x_{i}-\mu_{k})^{2}} \\
\end{aligned}$$

其中$N_{k}$为类别$k$出现的频率期望。

以上两步计算实质上对应了**期望最大化**(Expectation-Maximization)算法的**E步**(E-step)跟**M步**(M-step)。

## 多维数据时的情况

在多维数据下，需要为每个类生成一个多维高斯分布，表示方式与单维情况稍有不同：

$$
N(x_{i}|\mu_{k},\Sigma_{k})=\frac{1}{(2\pi)^{n/2}\Sigma_{k}^{1/2}}exp(-\frac{1}{2}(x_{i}-\mu_{k})^{T}\Sigma_{k}^{-1}(x_{i}-\mu_{k}))
$$

## 训练

有了算法框架，怎么训练模型呢。在初始时随机生成$$K$$个高斯分布，然后不断地迭代EM算法，直至似然函数变化不再明显或者达到了最大迭代次数。

### E-step

在给定的多维高斯分布下，计算各样本属于各个类别的概率：

$$
P(c_{k}|x_{i})=\frac{\pi_{k}P(c_{k}|x_{i})}{\sum\limits_{k=1}^{K}\pi_{k}P(c_{k}|x_{i})}
$$

### M_step

根据概率重新计算更优的高斯参数：

$$
\begin{aligned}
    N_{k}&=\sum\limits_{x=1}^{N}P(c_{k}|x_{i}) \\
    \pi_{k}&=\frac{N_{k}}{N} \\
    \mu_{k}&=\frac{1}{N_{k}}\sum\limits_{i=1}^{N}P(c_{k}|x_{i})x_{i} \\
    \Sigma_{k}&=\frac{1}{N_{k}}\sum\limits_{i=1}^{N}P(c_{k}|x_{i})(x_{i}-\mu_{k})^{T}(x_{i}-\mu_{k}) \\
\end{aligned}
$$