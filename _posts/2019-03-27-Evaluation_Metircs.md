---
layout: post
title:  "Evaluation Metrics"
categories: evaluation
tags: model eval metric
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

[代码](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/accuracy_score.py)

## Log-loss

对于二分类任务，特别是在logistic regression中，由最大似然法可以得出一个交叉熵损失函数：

$$
loss=-\sum\limits_{i}^{n}[y^{(i)}\ln(\hat{y}^{(i)})+(1-y^{(i)})\ln(1-\hat{y}^{(i)})]
$$

当然，作为损失函数，该值应该是越小越好。

[代码](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/log_loss.py)

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

由计算公式不难得出召回率的意义：针对数据中某一特定类别的样本，模型预测出了多少，相当于模型找出了多少。

## F1-score

可以看出无论是精确率还是召回率，它们都只是从不同的角度来衡量分类器的效果，前者从预测结果出发，后者从训练数据出发。那么如何设立指标来如何综合评价一个分类器的好坏？**F1分数**(F1-score)可以实现这个目的。

$$
F1=\frac{precision*recall}{precision + recall}
$$

由于precision与recall都是针对一个特定的类别计算的，所以F1分数有几个变种：

- micro-F1: 对所有类别的TP，FP，FN求和，使用加和的TP，FP，FN计算得到一个F1分数
- macro-F1: 分别计算所有类别下的F1分数，然后再计算平均F1分数
- weighted-F1: 分别计算所有类别下的F1分数，然后根据类分布概率计算加权平均F1分数

[代码](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/f1_score.py)

## ROC

**受试者工作特征曲线**(Receiver Operating Characteristic curve)，可用于评估二分类模型。首先明确几个概念：

- 假正例率(False Positive Rate)：计算方式为$FPR=\frac{FP}{N}$，意为模型把负样本预测为正样本的概率
- 真正例率(True Positive Rate)：计算方式为$TPR=\frac{TP}{P}$，意为模型把正样本预测为正样本的概率

ROC曲线即令FPR为横轴、TPR为纵轴绘制出的曲线。由上述定义不难看出，完美模型应当满足$FPR=0$而$TPR=1$。而实际上对于那些输出概率的二分类模型，如logistic regression，其预测的输出值是由设定的阈值决定的(通常情况下为0.5)。在不同的设定阈值下，二分类模型会有不同的FPR与TPR，那么通过不断设定不同的阈值，就会得到一系列FPR与TPR，即可绘制出该二分类模型的ROC曲线。

有了ROC曲线之后，就可以计算一个数值指标：曲线下面积(AUC)。可以证明，AUC的实际含义为：随机抽出一个正负样本对，模型对正样本的预测概率比负样本要大的概率：

$$
AUC=P(P_{pos}>P_{neg})
$$

那么对于AUC的计算，可以通过遍历所有样本对，然后计算出以上概率即可。有一个利用ranking性质的简便计算公式，假如总共有$n$个样本，其中$M$个正样本，$N$个负样本，首先按照模型对各样本的预测概率做排序。那么对于rank为$1$(概率最大)的正样本，跟它组合的所有$n-1$个样本的概率都要小于它，但是这$n-1$个样本中包含了$M-1$个正样本，需要排除。

对于rank为$1$的正样本，满足条件的正负样本对数为$n-1-(M-1)$；对于rank为$2$的正样本，满足条件的正负样本对数为$n-2-(M-2)$；以此类推，可以得到：

$$
AUC=P(P_{pos}>P_{neg})=\frac{\sum\limits_{i{\in}pos}reversed\_rank_{i}-\frac{M(M+1)}{2}}{M*N}
$$

其中$reversed\_rank_{i}$为逆序rank值，只是为了书写简便。

[实现指导](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/AUC.ipynb)

[完整代码](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/accuracy_score.py)

# Clustering

# Regression