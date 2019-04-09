---
layout: post
title:  "Sequential Minimal Optimization"
categories: optimization
tags: SVM optimization
---

* content
{:toc}

# 算法概述

在之前讲解SVM博客中，分析了SVM模型的理论基础与优化目标，并且讨论了SVM在达到最优解时的一些性质。但是前文中并没有提及SVM目标函数的优化方法，本文的目的就是讨论二次优化算法SMO用于SVM的学习。因为SMO算法涉及到的很多数学知识已超出本文范畴，某些地方只给出直接结论。

首先回顾SVM的优化目标为：

$$
\begin{aligned}
\min\limits_{\lambda}\ L(\lambda)&=\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\lambda_{i}\lambda_{j}y^{i}y^{j}x^{i}{x^{j}}^{T}-\sum_{i=1}^{m}\lambda_{i} \\
s.t. \ & 0\le\lambda_{i}\le{C}, \  \sum\limits\lambda_{i}y^{i}=0
\end{aligned}
$$

为了将核函数加入进来，将目标函数中两训练样本的内积替换成核函数的形式：

$$
\begin{aligned}
\min\limits_{\lambda}\ L(\lambda)&=\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\lambda_{i}\lambda_{j}y^{i}y^{j}\kappa_{ij}-\sum_{i=1}^{m}\lambda_{i} \\
s.t. \ & 0\le\lambda_{i}\le{C}, \  \sum\limits\lambda_{i}y^{i}=0
\end{aligned}
$$

SMO算法的核心思想是：每次只选取一对参数进行优化。假设在上述目标中，我们只令$\lambda_{a}$与$\lambda_{b}$为参数，其他$\lambda$为常数，那么优化问题可以写成：

$$
\begin{aligned}
\min\limits_{\lambda_{a},\lambda_{b}} & \frac{1}{2}\lambda_{a}^{2}{y^{(a)}}^{2}\kappa_{aa}+\frac{1}{2}\lambda_{b}^{2}{y^{(b)}}^{2}\kappa_{bb}+\frac{1}{2}\lambda_{a}y^{a}\sum\limits_{i{\ne}a}\lambda_{i}y^{i}\kappa_{ai}+\frac{1}{2}\lambda_{b}y^{b}\sum\limits_{i{\ne}b}\lambda_{i}y^{i}\kappa_{bi}-\lambda_{a}-\lambda_{b}-\sum\limits_{i{\ne}a,b}\lambda_{i} \\
s.t. \ & 0\le\lambda_{a,b}\le{C}, \  \lambda_{a}y^{a}+\lambda_{b}y^{b}=-\sum\limits_{i{\ne}a,b}\lambda_{i}y^{i} \\
\end{aligned}
$$

去除无关常量，简化后的优化目标可以写成：

$$
\begin{aligned}
\min\limits_{\lambda_{a},\lambda_{b}} & \frac{1}{2}\lambda_{a}^{2}\kappa_{aa}+\frac{1}{2}\lambda_{b}^{2}\kappa_{bb}+\frac{1}{2}\lambda_{a}y^{a}\sum\limits_{i{\ne}a}\lambda_{i}y^{i}\kappa_{ai}+\frac{1}{2}\lambda_{b}y^{b}\sum\limits_{i{\ne}b}\lambda_{i}y^{i}\kappa_{bi}-\lambda_{a}-\lambda_{b} \\
s.t. \ & 0\le\lambda_{a,b}\le{C}, \  \lambda_{a}y^{a}+\lambda_{b}y^{b}=-\sum\limits_{i{\ne}a,b}\lambda_{i}y^{i} \\
\end{aligned}
$$

在前文中提过SVM在优化后的一些性质，如对于分类正确的样本，其对应的$\lambda_{i}$是等于$0$的，同样的，那么对于软间隔SVM，不难推出优化后的几个性质：

|样本分类情况|对应的$\lambda$|
|-|-|
|$y^{i}(x^{i}\theta^{T}+\theta_{0}){\ge}1$|$\lambda_{i}=0$|
|$y^{i}(x^{i}\theta^{T}+\theta_{0}){\le}1$|$\lambda_{i}=C$|
|$y^{i}(x^{i}\theta^{T}+\theta_{0})=1$|$0<\lambda_{i}<C$|

## 优化策略

SMO每次只选取一对$\lambda$视为参数，假设先选定$\lambda_{a}$，那么$\lambda_{b}$的优化公式为：

$$
\begin{aligned}
    \lambda_{b}:&=\lambda_{b}-\frac{y^{b}((\hat{y}^{a}-y^{a})-(\hat{y}^{b}-y^{b}))}{2\kappa_{ab}-\kappa_{aa}-\kappa_{bb}} \\
    &=\lambda_{b}-\frac{y^{b}(E_{a}-E_{b})}{\eta} \\
\end{aligned}
$$

然后再看优化问题中的约束条件$\lambda_{a}y^{a}+\lambda_{b}y^{b}=-\sum\limits_{i{\ne}a,b}\lambda_{i}y^{i}$，由于只有$\lambda_{a}$与$\lambda_{b}$是参数，那么该优化条件还可以写成：$\lambda_{a}y^{a}+\lambda_{b}y^{b}=\xi$。而$y^{a}$与$y^{b}$的可能取值为$\{-1,+1\}$，由几何方法可以得到优化参数$\lambda$的一个上下界：

- 若$y^{a}{\ne}y^{b}$，$L=\max(0,\lambda_{b}-\lambda_{a})$，$H=\min(C,C+\lambda_{b}-\lambda_{a})$
- 若$y^{a}=y^{b}$，$L=\max(0,\lambda_{a}+\lambda_{b}-C)$，$H=\min(C,\lambda_{b}+\lambda_{a})$

所以，在优化之后，还需要检验$\lambda_{b}$是否还符合约束条件，若不满足，则需要做截断处理：

$$
\lambda_{b}=
\begin{cases}
    H & \text{if $\lambda_{b}>H$} \\
    \lambda_{b} & \text{if $L<\lambda_{b}<H$} \\
    L & \text{if $\lambda_{b}<L$} \\
\end{cases}
$$

而$\lambda_{a}$的优化公式为：

$$
\lambda_{a}:=\lambda_{a}-y^{a}y^{b}\Delta\lambda_{b}
$$

其中$\Delta\lambda_{b}=\lambda_{b}^{new}-\lambda_{b}^{old}$。

针对任一一个$\lambda$，若$\lambda$在边界范围$(0,C)$内，可以推出对应的$\theta_{0,k}$：

$$
\theta_{0,k}=\theta_{0}-E_{k}-y^{a}\Delta\lambda_{a}\kappa_{ak}-y^{b}\Delta\lambda_{b}\kappa_{kb} \qquad k={a,b}
$$

那么将其写成一个条件函数，可得到$\theta_{0}$的迭代优化公式：

$$
\theta_{0}:=
\begin{cases}
    \theta_{0,a} & \text{if $0<\lambda_{a}<C$} \\
    \theta_{0,b} & \text{if $0<\lambda_{b}<C$} \\
    (\theta_{0,a}+\theta_{0,b})/2 & \text{otherwise} \\
\end{cases}
$$

[实现指导](https://github.com/Daya-Jin/ML_for_learner/blob/master/svm/SMO.ipynb)
