---
layout: post
title:  "Sequential Minimal Optimization"
categories: optimization
tags: SVM optimization
---

* content
{:toc}

# 算法概述

在之前讲解SVM博客中，分析了SVM模型的理论基础与优化目标，并且讨论了SVM在达到最优解时的一些性质。但是前文中并没有提及SVM目标函数的优化方法，本文的目的就是讨论二次优化算法SMO用于SVM的学习。

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

（待补充）