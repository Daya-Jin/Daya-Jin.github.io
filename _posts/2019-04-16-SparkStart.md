---
layout: post
title:  "PySpark Start"
categories: pyspark
tags: pyspark bigdata
---

* content
{:toc}

# 概述

pyspark下的子模块主要有：
- ```pyspark.sql```：关于SQL和DataFrames的模块
- ```pyspark.streaming```：流式计算模块
- ```pyspark.ml```：基于DataFrame的机器学习模块

pyspark的初始代码一般为：

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local") \
    .appName("Demo") \
    .getOrCreate()

sc = spark.sparkContext    # 开启一个spark上下文会话
```

SparkContext是spark功能的主要入口，在代码中常表示为```sc```。```sc```对象的[常用方法](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.SparkContext)：

- ```parallelize```：以一个本地Python集合生成一个RDD对象，推荐使用```range```

## RDD

pyspark中的基础数据结构为**弹性分布式数据集**(Resilient Distributed Dataset)，RDD具有如下下特性：
- In-memory Computation：在内存中进行计算
- Lazy Evaluation：使用DAG保存操作，只在必要时才会做计算
- Immutability：RDD是Read-Only的
- Cacheable or Persistence：可存放在内存或硬盘中
- Partitioned：数据分布式存储在各节点中
- Fault Tolerance：分布式产生了容错性
- Coarse-grained Operations：RDD的操作是粗粒度(一批一批)的，并不是元素级的操作

使用序列数据构造一个rdd并查看前5个元素：

```python
rdd = sc.parallelize(range(10))
rdd.take(5)
```

> [0, 1, 2, 3, 4]

应用Python中的同名高阶函数：

```
rdd.map(lambda x: x*x).take(5)
```

> [0, 1, 4, 9, 16]

```
rdd.filter(lambda x: x > 5).take(5)
```

> [6, 7, 8, 9]

```
rdd.reduce(lambda x, y: x+y)
```

> 45

使用RDD的```collect()```方法可以将RDD转成Python数据类型：

```
rdd.collect()
```

> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

|[RDD常用方法](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD)||||||
|:-:|:-:|:-:|:-:|:-:|:-:|
|[collect](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.collect)|[count](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.count)|[distinct](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.distinct)|[filter](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.filter)|[first](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.first)|[flatMap](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.flatMap)|
|[groupBy](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.groupBy)|[join](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.join)|[map](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.map)|[max](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.max)|[mean](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.mean)|[min](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.min)|
|[randomSplit](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.randomSplit)|[reduce](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.reduce)|[sample](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.sample)|[sortBy](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.sortBy)|[stats](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.stats)|[stdev](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.stdev)|
|[sum](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.sum)|[take](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.take)|[variance](http://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.RDD.variance)||||

## DataFrame

pyspark另一种常用数据结构是DataFrame，而DF又可分为Row和Column。

读入文件生成DF的示例代码为：

```python
df = spark.read.csv('/home/hujinzhi/PySpark/dataset/train.csv', header=None, inferSchema=True)
```

只选择指定列，并显示前3行：

```Python
cols = ['Time', 'RoomArea', 'RoomDir', 'Bedroom',
        'Livingroom', 'Rental']    # 只选取部分列做演示
df = df.select(cols)
df.show(3)
```

```
+----+-----------+-------+-------+----------+-----------+ \
|Time|   RoomArea|RoomDir|Bedroom|Livingroom|     Rental| \
+----+-----------+-------+-------+----------+-----------+ \
|   2|0.020854022|     WS|      3|         2|3.904923599| \
|   3|0.010923535|     ES|      2|         1|2.546689304| \
|   3|0.010923535|     ES|      2|         1|2.546689304| \
+----+-----------+-------+-------+----------+-----------+ \
```

显示DF的列信息：

```python
df.printSchema()
```

```
root
 |-- Time: integer (nullable = true)
 |-- RoomArea: double (nullable = true)
 |-- RoomDir: string (nullable = true)
 |-- Bedroom: integer (nullable = true)
 |-- Livingroom: integer (nullable = true)
 |-- Rental: double (nullable = true)
```

对DF中的条目进行计数：

```python
df.count()    # 对行计数
```

> 196539

输出描述性统计信息：

```python
df.select(['Time','RoomArea','Bedroom','Rental']).summary().show()    # 描述统计信息
```

```
+-------+------------------+--------------------+------------------+-----------------+
|summary|              Time|            RoomArea|           Bedroom|           Rental|
+-------+------------------+--------------------+------------------+-----------------+
|  count|            196539|              196539|            196539|           196539|
|   mean|2.1152290385114405|0.013138849743341008| 2.236634968123375|7.949313378405461|
| stddev|0.7869801628627767|0.008103513291823544|0.8969612494208798|6.310608757211932|
|    min|                 1|                 0.0|                 0|              0.0|
|    25%|                 1|         0.009268454|                 2|      4.923599321|
|    50%|                 2|         0.012909633|                 2|       6.62139219|
|    75%|                 3|          0.01489573|                 3|      8.998302207|
|    max|                 3|                 1.0|                11|            100.0|
+-------+------------------+--------------------+------------------+-----------------+
```

条件筛选：

```python
df.where((df.RoomArea > 0.3) &
         (df.Time == 3)).select('Time', 'RoomArea', 'RoomDir', 'Rental').show(3)    # 条件筛选
```

```
+----+-----------+-------+-----------+
|Time|   RoomArea|RoomDir|     Rental|
+----+-----------+-------+-----------+
|   3|0.330354187|      W|5.602716469|
|   3|0.490897054|      S|8.896434635|
|   3|0.490897054|      S|8.896434635|
+----+-----------+-------+-----------+
```

|[DF常用方法](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)||||||
|:-:|:-:|:-:|:-:|:-:|:-:|
|[collect](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.collect)|[columns](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.columns)|[count](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.count)|[describe](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.describe)|[distinct](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.distinct)|[drop](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.drop)|
|[dropDuplicates](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.dropDuplicates)|[dropna](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.dropna)|[fillna](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.fillna)|[foreach](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.foreach)|[groupBy](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.groupBy)|[head](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.head)|
|[join](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.join)|[orderBy](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.orderBy)|[printSchema](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.printSchema)|[randomSplit](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit)|[select](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.select)|[show](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.show)|
|[sort](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sort)|[take](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.take)|[where](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.where)|[withColumn](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.withColumn)|[withColumnRenamed](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.withColumnRenamed)||

## SQL

pyspark同样还支持执行SQL语句去访问SQL数据结构。为了模拟从SQL表中读取数据，将一个DF转成一个临时表来做演示：

```python
df.registerTempTable('TMP')
```

使用```spark.sql()```方法来执行SQL语句，注意返回的是DF数据结构。

```python
spark.sql('SELECT * FROM TMP LIMIT 5').show()
spark.sql('SELECT MIN(Rental) FROM TMP').show()    # 查找Rental的最小值
# 查找最小房屋面积对应的样本
spark.sql('SELECT * FROM TMP WHERE RoomArea IN (SELECT MIN(RoomArea) FROM TMP)').show()
```

值得特别一提的是，在zeppelin环境中，当直接使用sql解释器提取数据时，zeppelin会默认提供一个可视化组件。

```
%sql
SELECT Bedroom,count(1) FROM TMP GROUP BY Bedroom
```

在可视化组件的```settings```中设置好```keys```与```values```，部分输出如下所示：

![](/img/2019-04-25_14-30-40.bmp)

