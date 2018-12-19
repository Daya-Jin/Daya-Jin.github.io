---
layout: post
title:  "NeuralNetworks(need fix)"
categories: DeepLearning
tags: DeepLearning
---

* content
{:toc}
# 神经网络(残缺文章，待修正)

## 模型结构

![](/img/20170919152906221.png)

上图是一个简单的神经网络，$$x_{i}$$为样本特征，$$\hat{y}$$为网络输出，个变量之间的关系满足：

$$
\begin{align}
a^{[1]}&=\sigma(\sum\limits_{i=1}^{3}\theta^{[0]}_{i}x_{i}+b^{[0]}) \\
\hat{y}&=g(\sum\limits_{i=1}^{4}\theta^{[1]}_{i}a^{[1]}+b^{[1]}) \\
\end{align}
$$

其中$$\sigma(x)$$称为**激活函数**(activation function)，$$g(x)$$为输出激活函数；输入数据$$X$$所在的位置称为**输出层**(input layer)，$$a_{i}^{j}$$所在的位置称为**隐藏层**(hidden layer)，输出预测结果的称为**输出层**(output layer)，图中每一个圆圈称为**神经元**(Neuron)。

最常见的激活函数为$$\sigma(x)=\frac{1}{1+e^{-x}}$$，即logistics regression中的sigmoid函数；而输出激活函数需要根据模型任务来定，回归任务下为$$g(x)=x$$，二分类任务下$$g(x)=\frac{1}{1+e^{-x}}$$，多分类任务下$$g(x)=softmax(这里待补充)$$；损失函数也由具体任务来定。



## 数学原理

更详细一点的来说，假设现在有两个样本构成的数据集$$X$$：

$$
X=
\left[
\begin{matrix}
 x_{1}^{[1]} & x_{2}^{[1]} & x_{3}^{[1]} \\
x_{1}^{[2]} & x_{2}^{[2]} & x_{3}^{[2]} \\
\end{matrix}
\right]
$$

对应的，第0层的权重系数矩阵为：

$$
\theta^{[0]}=
\left[
\begin{matrix}
 \theta^{[0]}_{1\_x_{1}} & \theta^{[0]}_{1\_x_{2}} & \theta^{[0]}_{1\_x_{3}} \\
 \theta^{[0]}_{2\_x_{1}} & \theta^{[0]}_{2\_x_{2}} & \theta^{[0]}_{2\_x_{3}} \\
 \theta^{[0]}_{3\_x_{1}} & \theta^{[0]}_{3\_x_{2}} & \theta^{[0]}_{3\_x_{3}} \\
 \theta^{[0]}_{4\_x_{1}} & \theta^{[0]}_{4\_x_{2}} & \theta^{[0]}_{4\_x_{3}} \\
\end{matrix}
\right]
$$

由$$A^{[1]}={\sigma}(X{\theta^{[0]}}^{T}+b^{[0]})$$得：

$$
\begin{align}
A^{[1]}&=
\left[
\begin{matrix}
\sigma(x^{[1]}{\theta^{[0]}_{1}}^{T}+b_{1}^{[0]}) & \sigma(x^{[1]}{\theta^{[0]}_{2}}^{T}+b_{2}^{[0]}) & \sigma(x^{[1]}{\theta^{[0]}_{3}}^{T}+b_{3}^{[0]}) & \sigma(x^{[1]}{\theta^{[0]}_{4}}^{T}+b_{4}^{[0]}) \\
\sigma(x^{[2]}{\theta^{[0]}_{1}}^{T}+b_{1}^{[0]}) & \sigma(x^{[2]}{\theta^{[0]}_{2}}^{T}+b_{2}^{[0]}) & \sigma(x^{[2]}{\theta^{[0]}_{3}}^{T}+b_{3}^{[0]}) & \sigma(x^{[2]}{\theta^{[0]}_{4}}^{T}+b_{4}^{[0]}) \\
\end{matrix} 
\right] \\
&=\left[
\begin{matrix}
a_{1\_x^{[1]}}^{[1]} & a_{2\_x^{[1]}}^{[1]} & a_{3\_x^{[1]}}^{[1]} & a_{4\_x^{[1]}}^{[1]} \\
a_{1\_x^{[2]}}^{[1]} & a_{2\_x^{[2]}}^{[1]} & a_{3\_x^{[2]}}^{[1]} & a_{4\_x^{[2]}}^{[1]} \\
\end{matrix}
\right]
\end{align}
$$

同样，第一层也有一个权重系数矩阵：

$$
\theta^{[1]}=
\left[
\begin{matrix}
 \theta^{[1]}_{1\_a_{1}} & \theta^{[1]}_{1\_a_{2}} & \theta^{[1]}_{1\_a_{3}} & \theta^{[1]}_{1\_a_{4}} \\
\end{matrix}
\right]
$$

易得在二分类任务下，输出$$\hat{Y}$$为：

$$
\begin{align}
\hat{Y}&=\sigma(A^{[1]}{\theta^{[1]}}^{T}+b^{[1]}) \\
&=\left[
\begin{matrix}
\sigma(a_{x^{[1]}}^{[1]}\theta^{[1]}+b^{[1]}) \\
\sigma(a_{x^{[1]}}^{[2]}\theta^{[1]}+b^{[1]}) \\
\end{matrix}
\right] \\
&=\left[
\begin{matrix}
\hat{y}^{[1]} \\
\hat{y}^{[2]} \\
\end{matrix}
\right] \\
\end{align}
$$

然后我们来试着做一下梯度下降优化。损失函数为：

$$
\begin{align}
L(Y,\hat{Y})&=\frac{1}{2}\sum\limits_{i=1}^{2}[-y^{[i]}{\ln}\hat{y}^{[i]}-(1-y^{[i]}){\ln}(1-\hat{y}^{[i]})] \\
&=\frac{1}{2}[-Y^{T}\ln\hat{Y}-(1-Y)^{T}\ln(1-\hat{Y})]
\end{align}
$$

为便于后面的计算，首先明确$$\sigma(x)=\frac{1}{1+e^{-x}}$$的导数：

$$
\begin{align}
\frac{\partial{\sigma(x)}}{\partial{x}}&=\frac{-1}{(1+e^{-x})^{2}}\cdot(-e^{-x}) \\
&=\frac{1}{1+e^{-x}}\cdot\frac{e^{-x}+1-1}{1+e^{-x}} \\
&=\frac{1}{1+e^{-x}}\cdot(1-\frac{1}{1+e^{-x}}) \\
&=\sigma(x)\cdot(1-\sigma(x)) \\
\end{align}
$$

首先即使算第一层参数$$\theta^{[1]}$$的梯度：

$$
\begin{align}
\frac{\partial{L}}{\partial{\theta^{[1]}}}&=\frac{\partial{L}}{\partial{\hat{Y}}}\cdot\frac{\partial{\hat{Y}}}{\partial{\theta^{[1]}}} \\
&=[-\frac{Y}{\hat{Y}}+\frac{1-Y}{1-\hat{Y}}]{\cdot}[(\hat{Y}\odot(1-\hat{Y}))^{T}A^{[1]}] \qquad \%此处除法表示元素相除 \\
\end{align}
$$

需要注意的是，上述公式算出来的梯度是一个$$(2\times4)$$的矩阵，每一行代表一个样本对于参数$$\theta^{[1]}$$的梯度，在做梯度下降法更新参数的时候需要考虑所有样本的梯度和，即：

$$
\theta:=\theta-\alpha\sum\limits_{i=1}^{N}\frac{\partial{L(y,\hat{y}_{i})}}{\partial{\theta}}
$$

那么损失函数在所有样本上的梯度应该写成：

$$
\begin{align}
\frac{\partial{L}}{\partial{\theta^{[1]}}}&= \\
&= \\
\end{align}
$$

<div style='display: none'>
哈哈我是注释，不会在浏览器中显示。
我也是注释。
</div>
