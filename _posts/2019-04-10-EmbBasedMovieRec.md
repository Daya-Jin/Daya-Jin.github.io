---
layout: post
title:  "Embedding Based Movie RecSys"
categories: RecSys
tags: RecSys DeepLearning Embedding
---

* content
{:toc}

# 概述

本文简单介绍使用embedding技术来实现一个电影推荐算法，当然，推荐算法背后的核心任务是评分预测，即回归问题。

既然是电影推荐，所用数据肯定是经典的MovieLens。MovieLens-1M数据包含三张表：
- ```users.dat```，6040名用户的信息，包含性别、年龄、职业、邮编等信息
- ```movies.dat```，3900部电影的信息，包含电影名、类别等信息
- ```ratings.dat```，评分数据，记录了用户对电影的评分信息

## 思路

对于用户数据，选择放弃邮编特征，因为这里假定用户对电影的评分与地域无关。然后将年龄与性别拼接构成一个组合特征，再对所有特征编码即可。

电影数据虽然原生只包含电影名与类别两个特征，但是可以从电影名中提取一个年份特征出来。同时电影名作为一个字符串，可以使用NLP相关技术来抽取特征。注意类别特征是一个多属性特征，必要时需要做截断或者填充。

评分数据没啥好说的，一个连接，所有用户特征与电影特征都需要拼接到该表上。

对特征编码之后，在网络的第一层设置并行的embedding层；embedding层后接的是并行的FC层，用于对各特征的embedding进行降维；然后对所有的user特征拼接起来做全连接得到user特征，movie的所有特征拼起来做全连接得到movie特征；对user特征与movie特征做点积得到预测评分。以上处理过程有两个特征是例外。电影标题特征是一个文本序列，得到嵌入后可以使用NLP技术来提取特征；另一个是年份特征，电影年份特征是一个non-categorical特征，因此不对年份特征做embedding，只对其做一个非线性变换。

整个模型框架如下图所示：

![](/img/EmbBasedMovieRec.svg)

## 预处理

### User.dat

用户信息表非常规整，所以需要做的处理也比较简单。假设用户对电影的评分不受地域影响，那么可以放弃邮编特征：

```python
user_df.drop(['Zip-code'], axis=1, inplace=True)
```

对特征进行整形编码：

```python
gender2id = {gender: idx
                for idx, gender in enumerate(sorted(user_df.loc[:, 'Gender'].unique()))}
age2id = {age: idx
            for idx, age in enumerate(sorted(user_df.loc[:, 'Age'].unique()))}
occupation2id = {occu: idx
                    for idx, occu in enumerate(sorted(user_df.loc[:, 'Occupation'].unique()))}

user_df.loc[:, 'Gender'] = user_df.loc[:, 'Gender'].map(gender2id)
user_df.loc[:, 'Age'] = user_df.loc[:, 'Age'].map(age2id)
user_df.loc[:, 'Occupation'] = user_df.loc[:, 'Occupation'].map(occupation2id)
```

将年龄与性别拼接形成一个组合特征：

```python
user_df.loc[:, 'Age_Gender'] = user_df.loc[:, 'Age'].map(
    str) + user_df.loc[:, 'Gender'].map(str)
agegen2id = {agegen: idx
                for idx, agegen in enumerate(sorted(user_df.loc[:, 'Age_Gender'].unique()))}
user_df.loc[:, 'Age_Gender'] = user_df.loc[:, 'Age_Gender'].map(agegen2id)

user_df.drop(['Age', 'Gender'], axis=1, inplace=True)
```