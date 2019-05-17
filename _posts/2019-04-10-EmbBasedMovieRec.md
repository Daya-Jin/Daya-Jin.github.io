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

### users.dat

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

### movies.dat

提取年份特征：

```python
movie_df.loc[:, 'Year'] = movie_df.loc[:, 'Title'].str.extract('({})'.format(
    '\([\d]{4}\)'), expand=False).str.replace('[\(\)]', '').astype('int32')  # 提取年份
movie_df.loc[:, 'Title'] = movie_df.loc[:, 'Title'].str.replace(
    '\ \([\d]{4}\)', '')  # 删除年份
```

把类别特征切割成列表，同时做规整化处理：

```python
genre_set = set()
for genre in movie_df.loc[:, 'Genres'].str.split('|'):
    genre_set.update(genre)
genre_set.add('<PAD>')

# 类别转id
genre2id = {genre: idx for idx, genre in enumerate(genre_set)}
# 多类别转list
genres2list = {genres: [genre2id[genre] for genre in genres.split('|')]
                for idx, genres in enumerate(movie_df.loc[:, 'Genres'].unique())}

# 规整化处理
genres_max = 5  # 设定最大允许的长度
for genres in genres2list.keys():
    genres2list[genres] = genres2list[genres][:genres_max]  # 超出长度做截断
    for pad_num in range(genres_max - len(genres2list[genres])):  # 需要填充的数量
        genres2list[genres].append(genre2id['<PAD>'])
```

对电影标题做同样的处理：

```python
# 单词集
word_set = set()
for word in movie_df.loc[:, 'Title'].str.split():
    word_set.update(word)
word_set.add('<PAD>')
word2id = {word: idx for idx, word in enumerate(word_set)}
title2list = {title: [word2id[word] for word in title.split()]
                for idx, title in enumerate(movie_df.loc[:, 'Title'].unique())}

# 规整化处理
title_max = 8
for title in title2list.keys():
    title2list[title] = title2list[title][:title_max]
    for pad_num in range(title_max - len(title2list[title])):
        title2list[title].append(word2id['<PAD>'])
```

对所有特征编码：

```python
mid2id = {mid: idx
            for idx, mid in enumerate(sorted(movie_df.loc[:, 'MovieID'].unique()))}
year2id = {year: idx
            for idx, year in enumerate(sorted(movie_df.loc[:, 'Year'].unique()))}

movie_df.loc[:, 'MovieID'] = movie_df.loc[:, 'MovieID'].map(mid2id)
movie_df.loc[:, 'Title'] = movie_df.loc[:, 'Title'].map(title2list)
movie_df.loc[:, 'Genres'] = movie_df.loc[:, 'Genres'].map(genres2list)
movie_df.loc[:, 'Year'] = movie_df.loc[:, 'Year'].map(year2id)
```

### ratings.dat

表合并：

```python
rating_df.drop(['ts'], axis=1, inplace=True)
data = pd.merge(pd.merge(user_df, rating_df), movie_df)
```

### 分割与保存

```python
n_samples = len(data)
train_ratio = 0.8
cut_idx = int(n_samples * train_ratio)
train_df, test_df = data[:cut_idx], data[cut_idx:]
np.save('train.npy', train_df.values)
np.save('test.npy', test_df.values)
```

## 模型设计

### Embedding

设置三种嵌入维度：$32$、$16$、$8$，对各特征的嵌入维度如下表所示：

|Feature|Emb Size|
|:-:|:-:|
|u_id|$32$|
|u_occu|$8$|
|u_agegen|$8$|
|m_id|$32$|
|m_tit|$16$|
|m_year|$1$|
|m_gen|$8$|

以上特征中除了```m_tit```外，其他特征嵌入后得到的都应是一个一维向量。而```m_gen```因为是一个多值特征，一个样本中的每个genre都会得到一个嵌入向量，所以要做压缩，求和或均化。

```python
with tf.variable_scope('user_embedding', initializer=tf.random_uniform_initializer(-1.0, 1.0)):
    # u_id嵌入
    uid_emb_lookup = tf.get_variable('uid_embedding', [u_id_size, u_id_emb_size],
                                     dtype=tf.float32)
    uid_emb = tf.nn.embedding_lookup(uid_emb_lookup, u_id)

    # u_occu嵌入
    uoccu_emb_lookup = tf.get_variable('uoccu_embedding', [u_occu_size, u_occu_emb_size],
                                       dtype=tf.float32)
    uoccu_emb = tf.nn.embedding_lookup(uoccu_emb_lookup, u_occu)

    # u_age_gender嵌入
    uagegen_emb = tf.get_variable('uagegen_embedding', [u_agegen_size, u_agegen_emb_size],
                                  dtype=tf.float32)
    uagegen_emb = tf.nn.embedding_lookup(uagegen_emb, u_agegen)

with tf.variable_scope('movie_embedding', initializer=tf.random_uniform_initializer(-1.0, 1.0)):
    mid_emb_lookup = tf.get_variable('mid_embedding', [m_id_size, m_id_emb_size],
                                     dtype=tf.float32)
    mid_emb = tf.nn.embedding_lookup(mid_emb_lookup, m_id)

    mtit_emb_lookup = tf.get_variable('mtit_embedding', [m_voc_size, m_tit_emb_size],
                                      dtype=tf.float32)
    mtit_emb = tf.nn.embedding_lookup(mtit_emb_lookup, m_tit)

    mgen_emb_lookup = tf.get_variable('mgen_embedding', [m_gen_size, m_gen_emb_size],
                                      dtype=tf.float32)
    mgen_emb = tf.nn.embedding_lookup(mgen_emb_lookup, m_gen)
    mgen_emb = tf.reduce_mean(mgen_emb, axis=1)    # 查找得到的多重emb做平均
```

### TextCNN

TextCNN使用三种不同尺寸的核：

|Kernel Size|Depth|
|:-:|:-:|
|$1$|$8$|
|$2$|$4$|
|$3$|$2$|

```python
# 文本卷积网络参数
n_txt_cnn_kernels = (8, 4, 2)
text_cnn_ksize = (1, 2, 3)

with tf.name_scope('TextCNN'):
    # 在最后增加一维，扩成四维向量(batch_size,tit_len,m_tit_emb_size,1)
    mtit_emb_exp = tf.expand_dims(mtit_emb, -1)
    
    layers = list()
    for i in range(len(n_txt_cnn_kernels)):
        conv = tf.layers.conv2d(mtit_emb_exp, filters=n_txt_cnn_kernels[i],
                                kernel_size=(
                                    text_cnn_ksize[i], m_tit_emb_size),
                                padding='same', activation=tf.nn.relu)
        # 一次性pooling
        pool = tf.layers.max_pooling2d(conv, pool_size=(m_tit_size-text_cnn_ksize[i]+1, 1),
                                       strides=(1, 1))
        layers.append(pool)

    tit_pool = tf.concat(layers, axis=3)
    tit_dropout = tf.layers.dropout(tf.layers.flatten(tit_pool),
                                    rate=0.2, training=is_training)
```

### FC

所有的embedding特征与CNN特征都经过一个FC层，目的是降维与特征提取，然后将用户(电影)端的所有FC输出拼接起来再经过一个FC层得到用户(电影)特征。

```python
# FC层
unit_fc1 = 1    # 电影年份是一个单独标量，同样经过一层FC
unit_fc2 = 16    # 各embedding特征经过的FC层
unit_fc3 = 128    # 单端特征的联合FC

with tf.name_scope('user_fc'):
    uid_fc = tf.layers.dense(uid_emb, unit_fc2, activation=tf.nn.relu,
                             name='uid_fc')
    uoccu_fc = tf.layers.dense(uoccu_emb, unit_fc2, activation=tf.nn.relu,
                               name='uoccu_fc')
    uagegen_fc = tf.layers.dense(uagegen_emb, unit_fc2, activation=tf.nn.relu,
                                 name='uagegen_fc')

    user_fc = tf.concat([uid_fc, uoccu_fc, uagegen_fc], axis=1)
    user_fc = tf.layers.dense(user_fc, unit_fc3, activation=tf.nn.relu)
    user_fc = tf.layers.dropout(user_fc, rate=0.3, training=is_training,
                                name='user_fc')

with tf.name_scope('movie_fc'):
    mid_fc = tf.layers.dense(mid_emb, unit_fc2, activation=tf.nn.relu)
    mgen_fc = tf.layers.dense(mgen_emb, unit_fc2, activation=tf.nn.relu)
    mtit_fc = tf.layers.dense(tit_dropout, unit_fc2, activation=tf.nn.relu)
    myear_fc = tf.layers.dense(tf.reshape(m_year, [-1, 1]), unit_fc1,
                               activation=tf.nn.relu) 
    myear_fc=tf.cast(myear_fc,dtype=tf.float32)    # 为了concat转换类型

    movie_fc = tf.concat([mid_fc,mgen_fc,mtit_fc,myear_fc],axis=1)
    movie_fc=tf.layers.dense(movie_fc,unit_fc3,activation=tf.nn.relu)
    movie_fc=tf.layers.dropout(movie_fc,rate=0.3,training=is_training)
```

### 输出

```python
unit_O = 1    # 输出一个分数

logits = tf.expand_dims(tf.reduce_sum(user_fc * movie_fc,
                                      axis=1), axis=1)    # 输出分数，把向量扩成矩阵
```

以上就是核心代码的展示与注解，完整代码[见此](https://github.com/Daya-Jin/DL_for_learner/tree/master/RecSys)。默认参数下5个epoch达到的验证MSE为0.83左右。

![](/img/2019-05-17_10-58-43.bmp)
