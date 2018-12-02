---
layout: post
title:  "Logistic Regression"
categories: MachineLearning
tags: LR
---

* content
{:toc}

# 模型概述

假定有一组数据$$X$$与$$Y$$，其中

$$
X=
\left[
\begin{matrix}
 x^{(1)} \\
x^{(2)} \\
 \vdots \\
 x^{(m)} \\
\end{matrix}
\right]
$$

$$X$$总共包含$$m$$条数据，而每条数据$$x^{(i)}$$又可表示为：

$$
x^{(i)}=
\left[
\begin{matrix}
 x^{i}_{0} & x^{i}_{1} & \cdots & x^{i}_{n}
\end{matrix}
\right]
$$

$$Y$$是一组向量，具体展开为：

$$
Y=
\left[
\begin{matrix}
 y^{(1)} \\
y^{(2)} \\
 \vdots \\
y^{(m)} \\
\end{matrix}
\right]
$$

假设二分类数据服从伯努利分布，特征条件概率服从高斯分布：

$$
P(y=1)=\phi \\
P(y=0)=1-\phi \\
P(x|y=1)=\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{(x-\mu_{1})^{2}}{2\sigma^{2}}] \\
P(x|y=0)=\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{(x-\mu_{0})^{2}}{2\sigma^{2}}] \\
$$

给定一个样本，模型的输出为：

$$
\begin{align}
P(y=1|x)&=\frac{P(y=1)P(x|y=1)}{P(y=0)P(x|y=0)+P(y=1)P(x|y=1)} \\
&=\frac{1}{1+\frac{P(y=0)P(x|y=0)}{P(y=1)P(x|y=1)}} \\
&=\frac{1}{1+\frac{1-\phi}{\phi}exp[-\frac{(x-\mu_{0})^{2}}{2\sigma^{2}}+\frac{(x-\mu_{1})^{2}}{2\sigma^{2}}]} \\
&=\frac{1}{1+\frac{1-\phi}{\phi}exp\frac{2(\mu_{0}-\mu_{1})x+\mu_{1}^{2}-\mu_{0}^{2}}{2\sigma^{2}}} \\
&=\frac{1}{1+exp(\frac{\mu_{0}-\mu_{1}}{\sigma^{2}}x+\frac{\mu_{1}^{2}-\mu_{0}^{2}}{2\sigma^{2}}+\ln(\frac{1-\phi}{\phi}))}
\end{align}
$$

令$$-a=\frac{\mu_{0}-\mu_{1}}{\sigma^{2}}$$，$$-b=\frac{\mu_{1}^{2}-\mu_{0}^{2}}{2\sigma^{2}}+\ln(\frac{1-\phi}{\phi})$$，得：

$$
P(y=1|x)=\frac{1}{1+e^{-(ax+b)}}
$$

引入**几率**(odds)概念：

$$
\begin{align*}
odds&=\frac{P(y=1|x)}{P(y=0|x)} \\
&=\frac{P(y=1|x)}{1-P(y=1|x)} \\
&=e^{ax+b}
\end{align*}
$$

两边同时取对数：

$$
\begin{align*}
\ln(\frac{P(y=1|x)}{P(y=0|x)})&=ax+b
\end{align*}
$$

由此引出Logistic Regression的概念，以线性回归去拟合一个**对数几率**(log-odds)，其模型表达式为：

$$
\begin{align*}
\hat{y}^{(i)}
 &= \sigma(\theta_{0}x^{(i)}_{0}+\theta_{1}x^{(i)}_{1}+...+\theta_{n}x^{(i)}_{n}) \\
 &= \sigma(x^{(i)}\theta^{T}) \\
\end{align*}
$$

其中，$$\sigma(x)$$为：

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

其图像为：

![](img/Logistic-curve.svg)

Logistic Regression实质上是将线性回归扩展到了分类任务上，并支持概率输出，其表达式为：

$$
\hat{y}=\frac{1}{1+e^{-(x\theta^{T})}}
$$

为了简便，上式省略了样本标号$$i$$，下同。然后经过一系列变换：

$$
\begin{align*}
& \hat{y}= \frac{1}{1+e^{-(x\theta^{T})}}= \frac{e^{x\theta^{T}}}{1+e^{x\theta^{T}}}\\
& \frac{1}{\hat{y}} = 1+\frac{1}{e^{x\theta^{T}}} \\
& \frac{1-\hat{y}}{\hat{y}} = \frac{1}{e^{x\theta^{T}}}
\end{align*}
$$

得：

$$
ln\frac{\hat{y}}{1-\hat{y}}=x^{(i)}\theta^{T}
$$

**注意**，由于$$\sigma(x)$$函数的作用，Logistic Regression的输出其实是一个概率，输入数据为正样本的概率，即：

$$
\begin{align*}
\hat{y}&=P(y=1|x;\theta) \\
1-\hat{y}&=P(y=0|x;\theta) \\
\end{align*}
$$

那么，参数$$\theta$$关于$$X$$的似然函数为：

$$
\begin{align*}
L(\theta|X) &= \prod_{i}P(y=0|x;\theta)\prod_{i}P(y=1|x;\theta) \\
			&= \prod_{i}\hat{y}^{y}(1-\hat{y})^{1-y}
\end{align*}
$$

其对数似然函数为：

$$
\begin{align*}
lnL(\theta|X) &= \sum_{i}[yln(\hat{y})+(1-y)ln(1-\hat{y})] \\
&= \sum_{i}[yln\frac{\hat{y}}{1-\hat{y}}+ln(1-\hat{y})] \\
&= \sum_{i}[y*x\theta^{T}-ln(1+e^{x\theta^{T}})]
\end{align*}
$$

我们需要最大化似然函数，那么等价的最小化损失函数为：

$$
Loss(\theta)=\sum_{i}[-y*x\theta^{T}+ln(1+e^{x\theta^{T}})]
$$

## 决策边界

由于Logistic Regression的输出是一个$$p(\hat{y}=1)$$的概率，那么对于二分类任务，模型对某一样本做出判别的依据就是一个概率阈值。假如概率阈值为0.5，则当模型输出$$f(x)>0.5$$时判为正样本，而当模型输出$$f(x)<0.5$$时判为负样本，此时模型的决策边界是啥呢？

回顾一下$$\sigma(x)$$的图像，$$\sigma(x)$$恰好经过$$(0, 0.5)$$这个点，并且是单增函数，那么可以看出，模型的决策边界为：

$$
x^{(i)}\theta^{T}=\sigma(0.5)
$$

当然，决策边界会根据自定义阈值而改变。
