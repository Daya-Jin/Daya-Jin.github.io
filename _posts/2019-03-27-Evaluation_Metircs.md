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
precision=\frac{TP}{TP+FP}=\frac{TP}{P'}
$$

其中$P'$为预测为正例的样本数。由计算公式不难得出精确率的意义：模型预测结果中某一指定类的准确率。

## Recall

**召回率**(recall)是衡量分类器另一性能的一种指标，其计算公式为：

$$
recall=\frac{TP}{TP+FN}=\frac{TP}{P}
$$

其中$P$为数据中的正例数量。由计算公式不难得出召回率的意义：针对数据中某一特定类别的样本，模型预测出了多少，相当于模型找出了多少。

可以发现，精确率与召回率其实是从不同角度衡量了模型对以特定类别的准确率。假设有一个包含若干黑白小球的黑箱，我们关注的是某一特定颜色的小球如白色小球。那么抓出一把小球后，手上会有若干白色小球和黑色小球，那么精确率就等于手上白色小球占手上所有球的比率；而召回率等于手上白色小球占所有白色小球的概率。精确率是从预测的角度来评估模型，而召回率是从数据的角度来评估模型。

Precision与Recall的缺陷在于，两者都是对正样本预测性能的单方面评估。如果要使精确率尽量高，那模型可以选择只预测把握最大的正样本，其余全部判为负样本，那么精确率就是$\frac{1}{1}=100\%$；类似的，如果要使召回率尽量高，那模型可以全部预测为正样本，那么召回率就是$\frac{P}{P}=100\%$。

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

## PR曲线

以reacall为横轴，以precision为纵轴绘制的曲线。PR曲线只关注对正例的预测性能，上面提到的precision与recall的性质，可以发现只要是非完美模型，那么模型的这两个指标一定是互相矛盾的：如果需要最高的精确率，那么召回率就会变低；如果要最高的召回率，那么精确率就会降低。在满足不同要求的precision下，会有不同的recall，调整模型的概率判别阈值，就可以得到不同的recall、precision二元组，由此可以绘制PR曲线。

PR曲线的两个维度针体现的都是正样本的预测性能，所以数据类别分布对模型的影响会体现在PR曲线上。

## ROC

**受试者工作特征曲线**(Receiver Operating Characteristic curve)，可用于评估二分类模型。首先明确几个概念：

- 假正例率(False Positive Rate)：假正例即误把负样本预测成了正样本，计算方式为$FPR=\frac{FP}{N}$，意为模型把负样本预测为正样本的概率
- 真正例率(True Positive Rate)：真正例即预测正确的正样本，计算方式为$TPR=\frac{TP}{P}$，意为模型把正样本预测为正样本的概率。不难发现TPR=recall

ROC曲线即令FPR为横轴、TPR为纵轴绘制出的曲线。由上述定义不难看出，完美模型应当满足$FPR=0$而$TPR=1$，所以ROC曲线的$(0,1)$点代表着完美模型。注意到在ROC曲线的左下-右上对角线上，该对焦线上的点满足FPR=TPR，即一个样本会被等概率地预测为正负样本，即对角线代表了随机模型。对角线左上角的点满足TPR>FPR，代表着优于随机猜的模型，而右下角则反之。

实际上对于那些输出概率的二分类模型，如logistic regression，其预测的输出值是由设定的阈值决定的(通常情况下为0.5)。在不同的设定阈值下，二分类模型会有不同的FPR与TPR，那么通过不断设定不同的阈值，就会得到一系列FPR与TPR，即可绘制出该二分类模型的ROC曲线。一个好的模型应该满足模型点始终在左下-右上对角线的左上方，同时越接近$(0,1)$点越好。

注意到当模型无脑输出正样本时，FPR=TPR=1；当模型无脑输出负样本时，FPR=TPR=0。除此之外，当正负样本不平衡时ROC无法给出一个客观评价，假如$N>>P$，若好几个负样本被误判为正样本，此时TPR(recall)是不变的，并且FPR的变化并不明显，因为分母$N$太大了。在正例远远小于负例且我们只关注正例预测性能的情况下，ROC并不适合作为评测指标。不过这同时也是ROC的优点，比较稳定。

## AUC

有了ROC曲线之后，就可以计算一个数值指标：曲线下面积(AUC)。可以证明，ROC的AUC实际含义为：随机抽出一个正负样本对，模型对正样本的预测概率比负样本要大的概率：

$$
AUC=P(P_{pos}>P_{neg})
$$

那么对于AUC的计算，可以通过遍历所有样本对，然后计算出以上概率即可。有一个利用ranking性质的简便计算公式，假如总共有$n$个样本，其中$M$个正样本，$N$个负样本，首先按照模型对各样本的预测概率做排序。那么对于rank为$1$(概率最大)的正样本，跟它组合的所有$n-1$个样本的概率都要小于它，但是这$n-1$个样本中包含了$M-1$个正样本，需要排除。

对于rank为$1$的正样本，满足条件的正负样本对数为$n-1-(M-1)$；对于rank为$2$的正样本，满足条件的正负样本对数为$n-2-(M-2)$；以此类推，可以得到：

$$
AUC=P(P_{pos}>P_{neg})=\frac{\sum\limits_{i{\in}pos}reversed\_rank_{i}-\frac{M(M+1)}{2}}{M*N}
$$

其中$reversed\_rank_{i}$为逆序rank值，只是为了书写简便。可以看出，计算AUC时需要对模型输出的概率排序，从这个角度来看，ROC的曲线面积体现了模型把正样本排在负样本之前的概率。

[实现指导](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/AUC.ipynb)

[完整代码](https://github.com/Daya-Jin/ML_for_learner/blob/master/metrics/accuracy_score.py)

# Clustering

# Regression