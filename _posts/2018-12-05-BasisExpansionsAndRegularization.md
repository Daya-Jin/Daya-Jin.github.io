---
layout: post
title:  "Basis Expansions and Regularization"
categories: MachineLearning
tags: regression
---

* content
{:toc}

## Basis Expansions and Regularization

首先，假定数据$$X$$有$$P$$个特征，那么具有$$M$$项的广义线性模型可以表示成：

$$
\begin{align}
f(X)&=\sum\limits_{m}^{M}\theta_{m}h_{m}(X) \\
&=\theta_{1}h_{1}(X)+\cdots+\theta_{M}h_{M}(X) \\
\end{align}
$$

其中$$h_{m}(X)$$是对$$X$$某一特征或某几个特征的变换函数，通常取值如下所示：

- $$h_{m}(X)=X_{m}$$，直接取出$$X$$的第$$m$$个特征；特别地，当$$M=P$$时，此变换下的模型相当于简单线性模型
- $$h_{m}(X)=X_{i}^{2}$$，对$$X$$的某一列做平方变换
- $$h_{m}(X)=X_{i}X_{j}$$，对$$X$$的某两列做交互变换
- $$h_{m}(X)=\log(X_{i})$$，对数变换
- $$h_{m}(X)=\sqrt{X_{i}}$$，开方变换
- $$h_{m}(X)=I(L_{m}{\le}X_{i}<U_{m})$$，指示$$X$$的某一列是否属于指定区间



### 分段与子条

