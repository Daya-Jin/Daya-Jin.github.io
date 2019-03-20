---
layout: post
title:  "Data Scaling and Transformation"
categories: Data
tags: preprocess
---

* content
{:toc}

# Scaling

数据**缩放**的实质就是对数据进行无量纲化处理或弱化量纲，下面介绍几种常用的缩放方式。

## Linear Scale

**归一化**(Normalization)通常指把数据缩放到$[0,1]$区间或$[-1,1]$区间，其转换公式分别为：

$$
\begin{aligned}
    x&=\frac{x-x_{min}}{x_{max}-x_{min}} \\
    x&=\frac{x-\frac{1}{2}(x_{max}+x_{min})}{x_max-x_min} \\
\end{aligned}
$$

**标准化**(Standardization)的实质就是计算Z-分数(Z-score)：

$$
x=\frac{x-\mu}{\sigma}
$$

标准化后的数据服从标准正态分布。

分别对归一化与标准化的式子做一下变形：

$$
\begin{aligned}
    x_{norm}&=\frac{x-x_{min}}{x_{max}-x_{min}} \\
    &=\frac{1}{x_{max}-x_{min}}x-\frac{x_{min}}{x_{max}-x_{min}} \\
    x_{z}&=\frac{x-\mu}{\sigma} \\
    &=\frac{1}{\sigma}x-\frac{\mu}{\sigma} \\
\end{aligned}
$$

可以看出归一化与标准化实质上都相当于对数据的一个线性变换，只不过是线性变换的系数不同。由此可以探究两者之间的区别。

首先不难看出归一化的缩放系数只由数据中的两个值决定：$x_{min}$与$x_{max}$，这一特性就决定了归一化变换是不稳定的，它容易被异常值或离群值影响。并且归一化的输出范围固定为$[0,1]$或者$[-1,1]$。

而反观标准化，它的线性变换系数是由数据统计量$\mu$与$\sigma$决定的，不难看出当$\sigma>1$时标准化会缩小数据的分布，而当$sigma<1$的时候会放大数据的分布，总而言之标准化就会使得变换后的数据呈一个固定的分布状态。标准化并没有对变换后的数据范围作规定，它只保证数据整体的分布。

## Non-Linear Scale

当数据的取值跨度非常大时，考虑使用对数变换来缩小数据在量级上的差距。常用的对数变换有：

$$
\begin{aligned}
    x=\log_{2}(x+1) \\
    x=\log_{10}(x+1) \\
\end{aligned}
$$

除了对数变换外，还可以使用开方变换：

$$
x=\sqrt[p]{x}
$$

# Transformation
