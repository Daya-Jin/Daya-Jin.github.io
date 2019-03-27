---
layout: post
title:  "Evaluation Metrics"
categories: evaluation
tags: mdoel eval metric
---

* content
{:toc}

# 概述

前文已经提过评估模型的一些方法，包括偏差方差分解以及交叉验证等。那么在模型评估时更具体的一些指标有哪些呢？

# Classification

## Accuracy

**准确率**(accuracy)是在分类任务中最常用的一种性能度量指标，准确率逐一对比真实值与预测值是否相等，对相等的值进行计数，求出真实值与预测值相等的一个比率。计算公式如下：

$$
acc=\frac{1}{n}\sum\limits_{i=1}^{n}I(y^{(i)}=\hat{y}^{(i)})
$$

## Log-loss

对于二分类任务，特别是在logistic regression中，由最大似然法可以得出一个交叉熵损失函数：

$$
loss=-\sum\limits_{i}^{n}[y^{(i)}\ln(\hat{y}^{(i)})+(1-y^{(i)})\ln(1-\hat{y}^{(i)})]
$$

当然，作为损失函数，该值应该是越小越好。

## Precision

**精确率**(precision)是衡量分类器性能的另一指标，但是该指标只关注模型对某一特定类别的分类性能。假设我们关注的类别为$1$，首先引入几个概念：

- True Positive: 模型预测为$1$，正确标签也为$1$，即该标签下预测正确的样本数
- False Positive: 模型预测为$1$，正确标签不为$1$，即该标签下预测错误的样本数
- False Negative: 模型预测不为$1$，正确标签为$1$，即该标签下漏预测的样本数

精确率的计算公式为：

$$
precision=\frac{TP}{TP+FP}
$$

由计算公式不难得出精确率的意义：模型预测结果中某一指定类的准确率。

## Recall

**召回率**(recall)是衡量分类器另一性能的一种指标，其计算公式为：

$$
recall=\frac{TP}{TP+FN}
$$

由计算公式不难得出召回率的意义：针对数据中某一特定类别的样本，模型预测出了多少。

## F1-score

可以看出无论是

# Clustering

# Regression