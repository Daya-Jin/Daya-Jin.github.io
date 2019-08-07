---
layout: post
title:  "Statistics Base"
categories: Math
tags: math statistics
---

* content
{:toc}

# 概述

统计学的本质就是通过**观测数据**(data)去推断**整体数据**(population)的性质。

## Central Tendency

**Mean**，群体均值：

$$\mu=\frac{1}{N}\sum\limits_{i}^{N}x_{i}$$

样本均值：

$$\bar{x}=\frac{1}{n}\sum\limits_{i}^{n}x_{i}$$

后者是前者的一个估计。当数据中有离群值时，平均值会被影响。

**Median**，中位数，将观测样本排序后位于中间位置的值或值的均值。中位数不受离群值的影响。

**Mode**，众数，观测样本中出现次数最多的值。众数不受离群值的影响。

**Expected Value**，期望值，假设随机变量$X$的概率密度函数为$f(x)$，期望值为：

$$E(X)=\int_{-\infty}^{+\infty}x_{i}f(x_{i})\, dx$$

若$X$是离散的，则期望值为：

$$E(X)=\sum\limits_{i}^{N}x_{i}P(x_{i})$$

期望值一般记为：$\mathbb{E}_{x\sim{f(x)}}$。

**Z-score**，Z分数，表征样本与均值偏离了几个标准差：

$$z=\frac{x-\mu}{\sigma}$$

## Dispersion

**Variance**，群体方差：

$$\sigma^{2}=\frac{\sum_{i}^{N}(x_{i}-\mu)^{2}}{N}$$

样本方差：

$$S^{2}=\frac{\sum_{i}^{n}(x_{i}-\bar{x})^{2}}{n-1}$$

后者是前者的一个估计。

**Standard deviation**，群体标准差：$\sigma$，样本标准差：$s$。

## Distribution

**Gaussian Distribution**，高斯分布：

$$
\mathcal{N}(\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}exp\Big(-\frac{(x-\mu)^{2}}{2\sigma^{2}}\Big)
$$

对高斯分布而言，有一个$3\sigma$原则，即偏离均值超过$3$个$\sigma$的数据($z>3$)会被视为离群值。

**Bernoulli Distribution**，伯努利分布，也称二项分布：

$$
\begin{cases}
    P(X=1)=p \\
    P(X=0)=1-p \\
\end{cases}
$$

最经典的二项分布事件是抛硬币。更常用的是$n$重伯努利分布，表示做$n$次独立伯努利事件，某一事件发生$k$次的概率为：

$$
P(X=k)=C_{n}^{k}p^{k}(1-p)^{n-k}
$$

## Law

**Law of Large Numbers**，大数定律

**Central limit theorem**，中心极限定理，对一个群体不断抽样，对样本计算$\bar{x}$，重复多次后$\bar{x}$的频数服从$\mathcal{N}(\mu,\frac{\sigma}{\sqrt{n}})$，其中$\mu$为群体均值，$\sigma$为群体方差，$n$为样本容量。

## Test

**Hypothesis Test**，假设检验，首先对群体性质做一个期望不成立的空假设$H_{0}$，然后计算样本的统计量来决定是否拒绝空假设，假设检验即是反证法。

定义一个**P值**(P-value)，其等于在$H_{0}$成立时观测样本满足某一性质的概率：

$$
p=P(stas|H_{0})
$$

定义**显著性水平**(Significance Level)，$\alpha$表示能接受的P值下限是多少。当$p<\alpha$时就拒绝空假设$H_{0}$。

**Z-test**

$$
Z=\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}
$$

**t-test**

$$
t=\frac{\bar{X}-\mu}{s/\sqrt{n}}
$$

**$\mathcal{X}^{2}$-test**

$$
\mathcal{X}^{2}=\frac{(n-1)s^{2}}{\sigma^{2}}
$$

**Error**

做检验肯定可能出现错误，根据判断的结果有两种错误：

||$H_{0}$ True|$H_{0}$ False|
|:-:|:-:|:-:|
|reject $H_{0}$|Type I error|Good|
|fail to reject $H_{0}$|Good|Type II error|

当原假设$H_{0}$成立时，但是却拒绝了$H_{0}$，则发生了**第一类错误**；若原假设$H_{0}$实际不成立，但是却接受了$H_{0}$，则发生了**第二类错误**。

