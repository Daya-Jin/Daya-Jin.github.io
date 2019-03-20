---
layout: post
title:  "The Similarity and Distance Measurements"
categories: Math
tags: similarity  distance
---

* content
{:toc}

# Distance

## Manhattan Distance

**曼哈顿距离**又称曼哈顿街区距离，在现实中的意义可以通过一个例子来说明。假如我们在城市中要由一个位置达到另一个位置，需要走多远的距离。虽然两点之间的距离是直线距离最短，但是因为城市规划是有路线限制的，我们只能沿着街道走。一般认为街道呈十字型设计，那么$A(a_{1},a_{2})$、$B(b_{1},b_{2})$两点之间的街区距离为:

$$
dist(A,B)=|a_{1}-b_{1}|+|a_{2}-b_{2}|
$$

下面给出两$n$维向量$\vec{a}$与$\vec{b}$的曼哈顿距离计算公式：

$$
dist(\vec{a},\vec{b})=\sum\limits_{i=1}^{n}|a_{i}-b_{i}|
$$

## Euclidean Distance

欧几里得距离是现实生活中最常用的一种距离计算方法，在数学上也被称为几何距离。欧氏距离计算的是两点之间在空间上的一个真实距离或最短距离。

$$
dist(\vec{a},\vec{b})=\sqrt{\sum\limits_{i=1}^{n}(a_{i}-b_{i})^{2}}
$$

## Minkowski Distance

不难发现，曼哈顿距离与欧氏距离分别相当于两向量相减然后再取一个$L1$范数与$L2$范数。那么将其扩展，就得到了**闵可夫斯基距离**(Minkowski Distance)。

$$
dist(\vec{a},\vec{b})=\sqrt[p]{\sum\limits_{i=1}^{n}(a_{i}-b_{i})^{p}}
$$

## Hamming distance

**海明距离**是在信息领域中用于对比信息差异的一种度量方法，它计算的是两个位串数据相异的位数：

$$
dist(s_{1},s_{2})=\sum\limits_{i=1}^{n}I(s_{1}[i]{\ne}s_{2}[i])
$$

其中$I(x)$为指示函数，当$x$成立时取值为$1$，否则为$0$。

# Similarity

## inner product

度量两向量之间的相似性，首先想到的应该是向量之间的内积：

$$
\vec{a}\cdot\vec{b}=\vec{a}\vec{b}^{T}=\sum\limits_{i=1}^{n}a_{i}b_{i}
$$

在欧几里得空间中，内积还可以表示成几何表达式：

$$
\vec{a}\cdot\vec{b}=|\vec{a}||\vec{b}|\cos{\theta}
$$

其中$\theta$为两向量的余弦夹角。

## Cosine

由上述启发不难想到，**夹角余弦**(cosine)可以用来度量两向量在方向上的相似度，两向量同向为$1$，反向则为$-1$。

$$
\cos<\vec{a},\vec{b}>=\frac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}
$$
