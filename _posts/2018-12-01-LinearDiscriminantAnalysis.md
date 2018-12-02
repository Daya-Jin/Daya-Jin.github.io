---
layout: post
title:  "Linear Discriminant Analysis"
categories: MachineLearning
tags: LDA
---

* content
{:toc}

## 模型概述

假设现在有一个单变量二分类问题，并且标签服从二项分布，特征条件概率服从高斯分布：

$$
P(y=1)=\phi \\
P(y=0)=1-\phi \\
P(x|y=1)=\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{(x-\mu_{1})^{2}}{2\sigma^{2}}] \\
P(x|y=0)=\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{(x-\mu_{0})^{2}}{2\sigma^{2}}] \\
$$

那么在给定样本的条件下，这两个类别发生的条件概率分别为：

$$
P(y=1|x)=\frac{P(y=1)P(x|y=1)}{P(y=0)P(x|y=0)+P(y=1)P(x|y=1)} \\
P(y=0|x)=\frac{P(y=0)P(x|y=0)}{P(y=0)P(x|y=0)+P(y=1)P(x|y=1)} \\
$$

两者之间的对数几率可以写成：

$$
\begin{align}
\log\frac{P(y=1|x)}{P(y=0|x)}&=\log\frac{P(y=1)}{P(y=0)}+\log\frac{P(x|y=1)}{P(x|y=0)} \\
&=\log\frac{\phi}{1-\phi}+\log\frac{exp[-\frac{(x-\mu_{1})^{2}}{2\sigma^{2}}]}{exp[-\frac{(x-\mu_{0})^{2}}{2\sigma^{2}}]} \\
&=\log\frac{\phi}{1-\phi}-\frac{(x-\mu_{1})^{2}}{2\sigma^{2}}+\frac{(x-\mu_{0})^{2}}{2\sigma^{2}} \\
&=\frac{\mu_{1}-\mu_{0}}{\sigma^{2}}{\cdot}x-\frac{\mu_{1}^{2}-\mu_{0}^{2}}{2\sigma^{2}}+\log\frac{\phi}{1-\phi}
\end{align}
$$

LDA对于某一样本的线性判别函数可写成：

$$
\delta_{1}(x)=\frac{\mu_{1}}{\sigma^{2}}{\cdot}x-\frac{\mu_{1}^{2}}{2\sigma^{2}}+\log{\phi} \\
\delta_{0}(x)=\frac{\mu_{0}}{\sigma^{2}}{\cdot}x-\frac{\mu_{0}^{2}}{2\sigma^{2}}+\log{1-\phi} \\
$$

LDA模型的预测输出为：

$$
f(x)=\arg\max\limits_{k}\delta_{k}(x)
$$

更一般的，实际问题是多变量多分类问题，那么特征条件概率分布的参数均为矩阵。假如数据$$X$$有$$p$$个特征，在$$y=k$$的条件下，特征条件概率可以写成：

$$
P(x|y=k)=\frac{1}{(2\pi)^{p/2}|\Sigma_{k}|^{1/2}}exp(-\frac{1}{2}(x-\mu_{k})^{T}\Sigma_{k}^{-1}(x-\mu_{k}))
$$

线性判别函数为：

$$
\delta_{k}(x)=x^{T}\Sigma^{-1}\mu_{k}-\frac{1}{2}\mu_{k}^{T}\Sigma^{-1}\mu_{k}+{\log}P(y=k)
$$
