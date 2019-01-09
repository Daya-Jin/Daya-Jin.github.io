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

其中$$A​$$和$$B​$$表示的是两个互斥事件，$$A​$$称为**前因**(antecedent)，$$B​$$称为**后果**(consequent)，上述关联规则表示$$A​$$会导致$$B​$$。具体地，在购物情形中，表示购买了$$A​$$的顾客也会购买$$B​$$，那么商场就可以把$$A​$$、$$B​$$放在一起或者是打包销售。关联规则的强度可以用它的**支持度**(support)和**置信度**(confidence)：

$$
S(A\Rightarrow{B})=P(AB) \\
C(A\Rightarrow{B})=P(B|A)
$$

在选取规则时通常会对这两个值设一个最低阈值最小支持度$$minsup$$和最小置信度$$minconf$$。注意由关联规则分析得出来的关联规则并不保证具有因果关系。