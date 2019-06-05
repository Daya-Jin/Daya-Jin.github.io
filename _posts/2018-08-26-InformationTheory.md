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

### Maximum Likelihood Estimation

如果已观测到事件$X$的一系列发生概率，求使得这一系列概率出现可能性最大的参数$\theta$，使用最大似然估计：

$$
\hat{\theta}=\arg\max\limits_{\theta}\prod{P(x_{i}\vert\theta)}
$$

其中$p(x_{i}\vert\theta)$为事件$x_{i}$在参数$\theta$下的发生概率。特别地，如果某条件概率为：

$$
\begin{aligned}
    p(y\vert{x};\theta)&\sim{\mathcal{N}(x\theta,\sigma^{2})} \\
    &=\frac{1}{\sigma\sqrt{2\pi}}\exp(\frac{-(y-x\theta)^{2}}{2\sigma^{2}}) \\
\end{aligned}
$$

参数$\theta$在已有观测样本$\{(x_{i},y_{i})\}$下的最大似然为：

$$
\begin{aligned}
    \hat{\theta}&=\arg\max\limits_{\theta}\prod{P(y_{i}|x_{i};\theta)} \\
    &=\arg\max\limits_{\theta}\sum{\log\frac{1}{\sigma\sqrt{2\pi}}\exp(\frac{-(y_{i}-x_{i}\theta)^{2}}{2\sigma^{2}})} \\
    &=\arg\max\limits_{\theta}\sum{[\log\frac{1}{\sigma\sqrt{2\pi}}-\frac{(y_{i}-x_{i}\theta)^{2}}{2\sigma^{2}}]} \\
    &=\arg\min\limits_{\theta}(y_{i}-x_{i}\theta)^{2}
\end{aligned}
$$

### Maximum A Posteriori

在最大似然估计中，对于参数$\theta$没有做任何假设，意味着$\theta$可以服从任何分布，只要能使得观测事件发生的概率最大即可。假如在某些情况下，参数$\theta$也是服从某一分布的，那么最大似然估计就不再适用于参数估计了，而应该使用最大后验概率：

$$
\begin{aligned}
    \hat{\theta}&=\arg\max\limits_{\theta}\prod{P(\theta|x_{i})} \\
    &=\arg\max\limits_{\theta}\prod{\frac{P(x_{i}\vert\theta)P(\theta)}{P(x_{i})}}
\end{aligned}
$$

看可以看出MAP引入了参数$\theta$的先验分布。