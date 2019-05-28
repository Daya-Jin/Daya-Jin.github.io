---
layout: post
title:  "Auto Encoder"
categories: DeepLearning
tags: autoencoder deeplearning
---

* content
{:toc}

# 概述

## Auto Encoder

待补充。。。

## Variational Auto-Encoder

**变分自编码器**(VAE)严格来说属于生成模型，它并不是直接学到一个数据的嵌入表示(embedding)，而是从数据中学到一个隐概率分布，在隐概率分布中进行采样即可以生成数据。更通俗一点来说，数据背后的隐变量可以看做是一个低维空间。

因为VAE牵扯到变分分析的知识，首先简单写一下前置知识：

KL-散度，用来衡量用一个分布去拟合另一个分布时损失的信息量，计算公式为：

$$
KL(p{\vert}{\vert}q)=-\sum\limits_{i=1}^{N}p(x_{i})\log\frac{q(x_{i})}{p(x_{i})}
$$

其中$N$为样本数。注意KL散度具有不对称性，即$KL(p{\vert}{\vert}q)\ne{KL(q{\vert}{\vert}p)}$，还有就是KL散度恒大于等于0。

令观测数据为$x$，假设变量$x$背后存在一个隐变量$z$，那么$x$的取值可以由$p(x{\vert}z)$来决定，如果我们要由观测数据$x$反推出背后隐变量$z$，需要求得：

$$
p(z{\vert}x)=\frac{p(x,z)}{p(x)}=\frac{p(x{\vert}z)p(z)}{p(x)}
$$

发现这个东西不知道怎么求，那么就可以使用一个近似分布$q(z{\vert}x)$来实现对$p(z{\vert}x)$的拟合，那么两分布之间的KL散度为：

$$
\begin{aligned}
    KL(q(z{\vert}x){\vert}{\vert}p(z{\vert}x))&=-\sum\limits_{z}q(z{\vert}x)\log\frac{p(z{\vert}x)}{q(z{\vert}x)} \\
    &=-\sum\limits_{z}q(z{\vert}x)\log\frac{\frac{p(x{\vert}z)p(z)}{p(x)}}{q(z{\vert}x)} \\
    &=-\sum\limits_{z}q(z{\vert}x)[\log\frac{p(x{\vert}z)p(z)}{q(z{\vert}x)}-\log{p(x)}] \\
    &=-\sum\limits_{z}q(z{\vert}x)\log{p(x{\vert}z)}-\sum\limits_{z}q(z{\vert}x)\log\frac{p(z)}{q(z{\vert}x)}+\log{p(x)}\sum\limits_{z}q(z{\vert}x) \\
    &=-\sum\limits_{z}q(z{\vert}x)\log{p(x{\vert}z)}+KL(q(z{\vert}x){\vert}{\vert}p(z))+\log{p(x)} \\
    &=-E_{z\sim{q(z{\vert}x)}}\log{p(x{\vert}z)}+KL(q(z{\vert}x){\vert}{\vert}p(z))+\log{p(x)} \\
\end{aligned}
$$

注意在给定观测数据$x$的条件下，$\log{p(x)}$是一个确定的值，即常数；而我们设一个近似分布$q(z{\vert}x)$的目的也很明确：$\min KL(q(z{\vert}x){\vert}{\vert}p(z{\vert}x))$，因为KL散度的非负性质，该目标可以转化成最大化一个下界：

$$
\max E_{z\sim{q(z{\vert}x)}}\log{p(x{\vert}z)}-KL(q(z{\vert}x){\vert}{\vert}p(z))
$$

将优化问题应用于AutoEncoder，就对应于使用encoder去拟合$q(z{\vert}x)$，使用decoder去拟合$p(x{\vert}z)$。

上述推导只是给出一个指导方针，要形成VAE还需要一些细节上的考量。

$p(z)$是一个未知分布，为了问题能够解决，对隐变量的分布做一个先验假设，假设$p(z)\sim{N(\mu,\sigma)}$，那么encoder需要学习的不是给定一个$x$下确定的一个隐变量$z$，而是隐变量的两个分布参数$\mu$与$\sigma$，隐变量$z$有多少维参数就有多少维。

另外，因为encoder的输出并不是$z$，而是$\mu$与$\sigma$，所以decoder的输入$z$需要使用参数来构成：

$$
z=\mu+\epsilon\times\sigma
$$

其中$\epsilon\sim{N(0,1)}$，$z\sim{N(\mu,\sigma)}$，通过上述公式就实现了一个服从特定分布的随机采样功能。这一技巧称为**重参数化**(reparameterization)，目的就是能够通过优化方法来不断调优分布的参数$\mu$与$\sigma$，而如果使用类似于```random.randn```这样的方法来实现采样，那么分布的参数是无法改变的。

损失函数为：

$$
\begin{aligned}
    Loss_{VAE}&=KL(q(z{\vert}x){\vert}{\vert}p(z))-E_{z\sim{q(z{\vert}x)}}\log{p(x{\vert}z)} \\
    &=\frac{1}{2}\sum(\mu^{2}+\sigma^{2}-\log\sigma^{2}-1)-\sum[x\log\hat{x}+(1-x)\log(1-\hat{x})]
\end{aligned}
$$
