---
layout: post
title:  "Information Theory"
categories: Math
tags: math information
---

* content
{:toc}

# 概述

## Information Theory

### Information

一个事件的信息量被定义为：

$$
I(x)=-\log{P(x)}
$$

易得信息量的大小和事件发生的概率成反比，当某事件绝对发生($P(x)$=1)时，该事件不含信息量。

### Entropy

熵可以衡量信息量的大小，定义为：

$$
\begin{aligned}
    H(x)&=-\mathbb{E}_{x\sim{P}}[\log{P(x)}] \\
    &=-\sum{P(x)}\log{P(x)} \\
\end{aligned}
$$

熵越大，说明事件越具有随机性，那么所包含的信息量就越大。特别地，当$\log$函数以$2$为底时，信息熵指示了编码事件所有信息所需要的编码长度。如*掷硬币*这一事件的信息熵为：

$$
\begin{aligned}
    H(coin)&=-P(head)\log{P(head)}-P(tail)\log{P(tail)} \\
    &=-\frac{1}{2}\log{\frac{1}{2}}-\frac{1}{2}\log{\frac{1}{2}} \\
    &=1
\end{aligned}
$$

由此得编码*掷硬币*这一事件只需要$1$位。

### Cross entropy

交叉熵可以用于衡量两个分布之间的差异性，定义为：

$$
\begin{aligned}
    H(P,Q)&=-\mathbb{E}_{x\sim{P}}[\log{Q(x)}] \\
    &=-\sum{P(x)}\log{Q(x)} \\
\end{aligned}
$$

信息熵表示使用自身分布来编码信息所需要的位数，而交叉熵表示用一个错误分布$Q$来编码真实分布$P$所需要的平均位数。

### KL Divergence

KL散度也称相对熵，是用于两个分布差异的方法之，其定义为：

$$
\begin{aligned}
    KL(P\vert\vert{Q})&=\mathbb{E}_{x\sim{P}}\log\frac{P(x)}{Q(x)} \\
    &=\sum{P(x)[\log{P(x)-\log{Q(x)}}]}
\end{aligned}
$$

注意KL散度具有不对称性。