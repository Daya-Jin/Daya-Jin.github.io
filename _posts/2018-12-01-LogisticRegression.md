---
layout: post
title:  "Logistic Regression"
categories: MachineLearning
tags: LR
---

* content
{:toc}

# 模型概述

假定有一组数据$X$与$Y$，其中

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

$X$总共包含$m$条数据，而每条数据$x^{(i)}$又可表示为：

$$
x^{(i)}=
\left[
\begin{matrix}
 x^{i}_{0} & x^{i}_{1} & \cdots & x^{i}_{n}
\end{matrix}
\right]
$$

$Y$是一组向量，具体展开为：

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
\begin{aligned}
P(y=1|x)&=\frac{P(y=1)P(x|y=1)}{P(y=0)P(x|y=0)+P(y=1)P(x|y=1)} \\
&=\frac{1}{1+\frac{P(y=0)P(x|y=0)}{P(y=1)P(x|y=1)}} \\
&=\frac{1}{1+\frac{1-\phi}{\phi}exp[-\frac{(x-\mu_{0})^{2}}{2\sigma^{2}}+\frac{(x-\mu_{1})^{2}}{2\sigma^{2}}]} \\
&=\frac{1}{1+\frac{1-\phi}{\phi}exp\frac{2(\mu_{0}-\mu_{1})x+\mu_{1}^{2}-\mu_{0}^{2}}{2\sigma^{2}}} \\
&=\frac{1}{1+exp(\frac{\mu_{0}-\mu_{1}}{\sigma^{2}}x+\frac{\mu_{1}^{2}-\mu_{0}^{2}}{2\sigma^{2}}+\ln(\frac{1-\phi}{\phi}))}
\end{aligned}
$$

令$-a=\frac{\mu_{0}-\mu_{1}}{\sigma^{2}}$，$-b=\frac{\mu_{1}^{2}-\mu_{0}^{2}}{2\sigma^{2}}+\ln(\frac{1-\phi}{\phi})$，得：

$$
P(y=1|x)=\frac{1}{1+e^{-(ax+b)}}
$$

引入**几率**(odds)概念：

$$
\begin{aligned}
odds&=\frac{P(y=1|x)}{P(y=0|x)} \\
&=\frac{P(y=1|x)}{1-P(y=1|x)} \\
&=e^{ax+b}
\end{aligned}
$$

两边同时取对数：

$$
\begin{aligned}
\ln(\frac{P(y=1|x)}{P(y=0|x)})&=ax+b
\end{aligned}
$$

由此引出Logistic Regression的概念，以线性回归去拟合一个**对数几率**(log-odds)，其模型表达式为：

$$
\begin{aligned}
\hat{y}^{(i)}
 &= \sigma(\theta_{0}x^{(i)}_{0}+\theta_{1}x^{(i)}_{1}+...+\theta_{n}x^{(i)}_{n}) \\
 &= \sigma(x^{(i)}\theta^{T}) \\
\end{aligned}
$$

其中，$\sigma(x)$为：

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

其图像为：

![](img/Logistic-curve.svg)

Logistic Regression实质上是将线性回归扩展到了分类任务上，并支持概率输出，其表达式为：

$$
\hat{y}=\frac{1}{1+e^{-(x\theta^{T})}}
$$

为了简便，上式省略了样本标号$i$，下同。然后经过一系列变换：

$$
\begin{aligned}
& \hat{y}= \frac{1}{1+e^{-(x\theta^{T})}}= \frac{e^{x\theta^{T}}}{1+e^{x\theta^{T}}}\\
& \frac{1}{\hat{y}} = 1+\frac{1}{e^{x\theta^{T}}} \\
& \frac{1-\hat{y}}{\hat{y}} = \frac{1}{e^{x\theta^{T}}}
\end{aligned}
$$

得：

$$
ln\frac{\hat{y}}{1-\hat{y}}=x^{(i)}\theta^{T}
$$

**注意**，由于$\sigma(x)$函数的作用，Logistic Regression的输出其实是一个概率，输入数据为正样本的概率，即：

$$
\begin{aligned}
\hat{y}&=P(y=1|x;\theta) \\
1-\hat{y}&=P(y=0|x;\theta) \\
\end{aligned}
$$

那么，参数$\theta$关于$X$的似然函数为：

$$
\begin{aligned}
L(\theta|X) &= \prod_{i}P(y=0|x;\theta)\prod_{i}P(y=1|x;\theta) \\
			&= \prod_{i}\hat{y}^{y}(1-\hat{y})^{1-y}
\end{aligned}
$$

其对数似然函数为：

$$
\begin{aligned}
lnL(\theta|X) &= \sum_{i}[yln(\hat{y})+(1-y)ln(1-\hat{y})] \\
&= \sum_{i}[yln\frac{\hat{y}}{1-\hat{y}}+ln(1-\hat{y})] \\
&= \sum_{i}[y*x\theta^{T}-ln(1+e^{x\theta^{T}})]
\end{aligned}
$$

我们需要最大化似然函数，那么等价的最小化损失函数为：

$$
Loss(\theta)=\sum_{i}[-y*x\theta^{T}+ln(1+e^{x\theta^{T}})]
$$

对于logistic regression，同样可以使用梯度下降法来优化参数$$\theta$$。注意sigmoid函数的导数：

$$
\begin{aligned}
\frac{\partial{\sigma(x)}}{\partial{x}}&=\frac{-1}{(1+e^{-x})^{2}}\cdot(-e^{-x}) \\
&=\frac{1}{1+e^{-x}}\cdot\frac{e^{-x}+1-1}{1+e^{-x}} \\
&=\frac{1}{1+e^{-x}}\cdot(1-\frac{1}{1+e^{-x}}) \\
&=\sigma(x)\cdot(1-\sigma(x)) \\
\end{aligned}
$$

那么在标量形式下，易推得损失函数关于参数$$\theta​$$的梯度为：

$$
\begin{aligned}
\frac{\partial{L}}{\partial{\theta}}&=-\frac{y}{\hat{y}}{\cdot}\frac{\partial{\hat{y}}}{\partial{\theta}}+\frac{1-y}{1-\hat{y}}\cdot{\frac{\partial{\hat{y}}}{\partial\theta}} \\
&=-\frac{y}{\hat{y}}{\cdot}\hat{y}(1-\hat{y}){\cdot{x}}+\frac{1-y}{1-\hat{y}}{\cdot}\hat{y}(1-\hat{y}){\cdot}x \\
&=(\hat{y}-y)x
\end{aligned}
$$

注意到logistic regression的梯度形式与linear regression是一样的，唯一的区别就在于$\hat{y}$的不同。

## 决策边界

由于Logistic Regression的输出是一个$p(\hat{y}=1)$的概率，那么对于二分类任务，模型对某一样本做出判别的依据就是一个概率阈值。假如概率阈值为0.5，则当模型输出$f(x)>0.5$时判为正样本，而当模型输出$f(x)<0.5$时判为负样本，此时模型的决策边界是啥呢？

回顾一下$\sigma(x)$的图像，$\sigma(x)$恰好经过$(0, 0.5)$这个点，并且是单增函数，那么可以看出，模型的决策边界为：

$$
x^{(i)}\theta^{T}=\sigma(0.5)
$$

当然，决策边界会根据自定义阈值而改变；除此之外，logistic regression也可以设置为输出连续的概率值。

# 后记

logistic regression本质上还是属于linear model的一种，那么linear model所具有的优点，logistic regression也是有的；对于缺点也同样成立。

由于logistic regression背后的概率思想，如果训练样本存在样本平衡性问题，那么就会对该模型的表现有很大的影响。直观一点来说，logistic regression的决策边界会受到样本分布密度的推挤，其决策边界会比较偏近于少数类。

logistic regression还有一个被讨论的点就是关于高维稀疏特征的。

1. 首先，logistic regression作为一个线性模型，将特征之间做组合形成新特征是增强线性模型对非线性数据拟合能力的必要手段之一；
2. 另外，线性模型计算简单，在高维特征下的速度也是可以接受的；
3. 线性模型的正则化是对各个特征的权重做惩罚，不会在某一特征上产生过拟合；
4. 最后，对特征的离散化，会增强模型对于更细粒度特征的学习能力。