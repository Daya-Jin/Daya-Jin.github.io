---
layout: post
title:  "Neural Networks"
categories: DeepLearning
tags: DeepLearning
---

* content
{:toc}

# 神经网络

## 模型结构

![](/img/20170919152906221.png)

上图是一个简单的神经网络，$x_{i}$为样本特征，$\hat{y}$为网络输出，个变量之间的关系满足：

$$
\begin{aligned}
a^{[1]}&=\sigma(\sum\limits_{i=1}^{3}\theta^{[0]}_{i}x_{i}+b^{[0]}) \\
\hat{y}&=g(\sum\limits_{i=1}^{4}\theta^{[1]}_{i}a^{[1]}+b^{[1]}) \\
\end{aligned}
$$

其中$\sigma(x)$称为**激活函数**(activation function)，$g(x)$为输出激活函数；输入数据$X$所在的位置称为**输出层**(input layer)，$a_{i}^{j}$所在的位置称为**隐藏层**(hidden layer)，输出预测结果的称为**输出层**(output layer)，图中每一个圆圈称为**神经元**(Neuron)。

最常见的激活函数为$\sigma(x)=\frac{1}{1+e^{-x}}$，即logistics regression中的sigmoid函数；而输出激活函数需要根据模型任务来定，回归任务下为$g(x)=x$，二分类任务下$g(x)=\frac{1}{1+e^{-x}}$，多分类任务下$g(x)=softmax(这里待补充)$；损失函数也由具体任务来定。

## 数学原理

保持与实现上的一致性，令

$$
a^{[1]}=\sigma(x\theta^{[0]}+b^{[0]})
$$

$x$的形状为$(1,3)$，$a^{[0]}$的形状为$(1,4)$，那么有矩阵乘法的性质得$\theta^{[0]}$的形状为$(3,4)$；

$$
\hat{y}=\sigma(a^{[1]}\theta^{[1]}+b^{[1]})
$$

$\hat{y}$的形状为$(1,1)$，所以$\theta^{[1]}$的形状为$(4,1)$。由矩阵乘法性质不难推出，若当前层的单元数为$n^{[i]}$，下层单元数为$n^{[i+1]}$，则当前层权重矩阵的形状为：

$$
dim(\theta^{[i]})=(n^{[i]},n^{[i+1]})
$$

整个网络的输出可以写成：

$$
\begin{aligned}
    z^{[1]}&=a^{[0]}\theta^{[0]}+b^{[0]} \\
    a^{[1]}&=\sigma(z^{[1]}) \\
    z^{[2]}&=a^{[1]}\theta^{[1]}+b^{[1]} \\
    a^{[2]}&=\sigma(z^{[2]}) \\
\end{aligned}
$$

以二分类为例，简单写下神经网络的反向传播过程。为便于后面的计算，先明确$\sigma(x)=\frac{1}{1+e^{-x}}$的导数：

$$
\begin{aligned}
\frac{\partial{\sigma(x)}}{\partial{x}}&=\frac{-1}{(1+e^{-x})^{2}}\cdot(-e^{-x}) \\
&=\frac{1}{1+e^{-x}}\cdot\frac{e^{-x}+1-1}{1+e^{-x}} \\
&=\frac{1}{1+e^{-x}}\cdot(1-\frac{1}{1+e^{-x}}) \\
&=\sigma(x)\cdot(1-\sigma(x)) \\
\end{aligned}
$$

首先，损失函数为：

$$
L=-y{\ln}a^{[2]}-(1-y){\ln}(1-a^{[2]})
$$

逐层对变量求导：

$$
\begin{aligned}
    {\Delta}a^{[2]}&=\frac{\partial{L}}{\partial{a^{[2]}}} \\
    &=-\frac{y}{a^{[2]}}+\frac{1-y}{1-a^{[2]}} \\
    {\Delta}z^{[2]}&={\Delta}a^{[2]}\cdot\frac{\partial{a^{[2]}}}{\partial{z^{[2]}}} \\
    &={\Delta}a^{[2]}{\cdot}a^{[2]}(1-a^{[2]}) \\
    &=a^{[2]}-y \\
    {\Delta}\theta^{[1]}&={\Delta}z^{[2]}\cdot\frac{\partial{z^{[2]}}}{\partial\theta^{[1]}} \\
    &={\Delta}z^{[2]}{\cdot}a^{[1]} \\
    {\Delta}b^{[1]}&={\Delta}z^{[2]}\cdot\frac{\partial{z^{[2]}}}{\partial{b^{[1]}}} \\
    &={\Delta}z^{[2]} \\
\end{aligned}
$$

更前一层的梯度为：

$$
\begin{aligned}
    {\Delta}a^{[1]}&={\Delta}z^{[2]}\cdot\frac{\partial{z^{[2]}}}{\partial{a^{[1]}}} \\
    &={\Delta}z^{[2]}\cdot\theta^{[1]} \\
    {\Delta}z^{[1]}&={\Delta}a^{[1]}\cdot\frac{\partial{a^{[1]}}}{\partial{z^{[1]}}} \\
    &={\Delta}z^{[2]}\cdot\theta^{[1]}{\cdot}a^{[1]}(1-a^{[1]}) \\
    {\Delta}\theta^{[0]}&={\Delta}z^{[1]}\cdot\frac{\partial{z^{[1]}}}{\partial\theta^{[0]}} \\
    &={\Delta}z^{[1]}{\cdot}a^{[0]} \\
    {\Delta}b^{[0]}&={\Delta}z^{[1]}\cdot\frac{\partial{z^{[1]}}}{\partial{b^{[0]}}} \\
    &={\Delta}z^{[1]}
\end{aligned}
$$

这是使用sigmoid函数为激活函数下二分类神经网络的梯度。其实如果在更深层的神经网络中推导的话，假设有$h$层隐藏层，那么除了最后一层隐藏层，前$h-1$层的梯度都可以写成递推表达式，因为最后一层隐藏层的梯度是由损失函数推出来的，而前$h-1$层的梯度都是由当前层的输出$a$推出来的。递推式展开可以写成累乘，那么累乘就会有一个问题：当神经网络层数过深并且每一层的梯度都小于1时，那么越前面层的梯度就会越小，若都大于1，则越前面层的梯度就会越大。这就是深层神经网络中**梯度消失**与**梯度爆炸**的问题。

[原博客的Python实现指导](https://blog.csdn.net/qq_31823267/article/details/78044065)

[tensorflow实现指导](https://github.com/Daya-Jin/DL_for_learner/blob/master/DNN/DNN.ipynb)