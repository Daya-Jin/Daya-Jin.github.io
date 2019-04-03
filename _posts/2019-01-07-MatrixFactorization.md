---
layout: post
title:  "Matrix Factorization"
categories: factorization
tags: matrix factorization
---

* content
{:toc}

# 概述

在机器学习领域通常会用到矩阵分解技术，目的就是维度规约或压缩存储，本文做一个简单的总结与概述。

# EVD

**特征值分解**(Eigenvalue Decomposition)，假设对于一个$n{\times}n$的方阵$A$，有如下等式成立：

$$
A\vec{v}=\lambda\vec{v}
$$

其中$\lambda$为常数，$\vec{v}$为列向量。那么满足上式的$\lambda$为矩阵$A$的特征值，对应的$\vec{v}$为特征向量，方阵的特征向量是相互正交的。写成矩阵形式有：

$$
A=Q{\Sigma}Q^{-1}
$$

其中$\Sigma$为特征值由大到小排列构成的对角矩阵，$Q$为特征向量构成的方阵。选取前$k$大的特征值，那么降维后的$A$可以表示成：

$$
A_{reduc}=A_{n{\times}n}(Q^{-1})_{n{\times}k}
$$

EVD即是PCA的原理。

# SVD

**奇异值分解**(Singular Value Decomposition)，假设对一个$n{\times}m$的矩阵$A$，SVD的目标是把$A$分解成如下形式：

$$
A=U{\Sigma}V^{T}
$$

其中$\Sigma$是与$A$同形状的奇异值矩阵。由矩阵乘法的性质可得，矩阵$U$的形状为$n{\times}n$，$V^{T}$的形状为$m{\times}m$。同样类似的，$U$与$V$都是正交方阵。

SVD可以通过EVD来实现，注意到：

$$
AA^{T}=U\Sigma\Sigma^{T}U^{T} \\
A^{T}A=V\Sigma^{T}{\Sigma}V^{T} \\
$$

不难发现可以分别通过对$AA^{T}$和$A^{T}A$做EVD可以得到$U$和$V$，而$\Sigma$则是特征值的开方。选取前$k$大的奇异值，那么$A$可以近似压缩存储成：

$$
A_{comp}=U_{n{\times}k}\Sigma_{k{\times}k}(V^{T})_{k{\times}m}
$$

对于降维，有：

$$
A_{reduc}=A_{n{\times}m}(V^{T})_{m{\times}k}
$$