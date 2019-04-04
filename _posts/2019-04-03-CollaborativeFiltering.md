---
layout: post
title:  "Collaborative Filtering"
categories: recommend
tags: recommend
---

* content
{:toc}

# 概述

**协同过滤**(Collaborative Filtering)是推荐系统中最经典的方法了，本文做一个简单的概述。

# User-based Collaborative Filtering

假设一个场景，我们想买一个东西或者想吃一个东西，但是自己不知道哪种东西比较好，那么通常的选择就是去询问身边有着相似喜好的朋友寻求推荐。这就是基于用户的协同过滤，核心思想就是相似的**用户**(user)会喜欢相似的**物品**(item)。

## 用户数据

为了在用户群体中找到跟自己相似的用户，很明显需要收集所有用户的数据，如所有用户对多个商品的评价，那么该数据的矩阵形状为$(n_{users},n_{items})$。在该矩阵中计算其他所有用户与指定用户的相似度，并使用前$k$个相似用户的数据来做推荐。