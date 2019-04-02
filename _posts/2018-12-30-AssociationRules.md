---
layout: post
title:  "Association Rules"
categories: MachineLearning
tags: RecommendSystem
---

* content
{:toc}
## 概述

在商场的购物数据中，常常可以看到多种物品同时出现，这背后隐藏着联合销售或打包销售的商机。**关联规则分析**(Association Rule Analysis)就是为了发掘购物数据背后的商机而诞生的。

定义一个关联规则：

$$
A\Rightarrow{B}
$$

其中$A​$和$B​$表示的是两个互斥事件，$A​$称为**前因**(antecedent)，$B​$称为**后果**(consequent)，上述关联规则表示$A​$会导致$B​$。具体地，在购物情形中，表示购买了$A​$的顾客也会购买$B​$，那么商场就可以把$A​$、$B​$放在一起或者是打包销售。关联规则的强度可以用它的**支持度**(support)和**置信度**(confidence)：

$$
S(A\Rightarrow{B})=P(AB) \\
C(A\Rightarrow{B})=P(B|A)=\frac{P(AB)}{P(A)}
$$

可以看出支持度即两个事件同时发生的概率，置信度即在前因发生的条件下，后果发生的概率。

在选取规则时通常会对这两个值设一个最低阈值最小支持度$min_{sup}$和最小置信度$min_{conf}$。注意由关联规则分析得出来的关联规则并不保证具有因果关系。

**项集**(itemset)被定义为包含$0$个或多个项的集合，支持度大于阈值$min_{sup}$的项集被称为**频繁项集**(frequent itemset)，频繁项集中置信度大于阈值$min_{conf}$的规则称为**强规则**(strong rule)。关联规则的目的就是找到频繁项集与强规则。

由概率出发不难得到关于频繁项集的一个性质：频繁项集的所有子集都是频繁的，即$P(A){\ge}P(AB){\ge}min_{sup}$；非频繁项集的超集都是非频繁的，即$min_{sup}{\ge}P(A){\ge}P(AB)$。这一性质能大大减少搜索频繁项集时的搜索空间。