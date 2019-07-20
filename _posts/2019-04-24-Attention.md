---
layout: post
title:  "Attention"
categories: NLP
tags: NLP 
---

* content
{:toc}

## Seq2Seq

讲到Attention就不得不提NLP领域最经典的Seq2Seq模型架构：

![](/img/seq2seq_ts.png)

Seq2Seq架构分为两部分，前一部分是**编码器**(Encoder)，后一部分是**解码器**(Decoder)。以RNN为例(不限于RNN)，作为Encoder的RNN负责读入全部输入句子$X$，得到一个**上下文向量**(Context Vector)。Decoder的作用就是读入这个Context Vec，然后逐步预测$Y$。Seq2Seq架构的最大缺点就是把整个序列都压缩到了一个Context Vec中，Decoder要想从单个向量中精确并完整地预测时序信息比较难。

## Attention

Bahdanau在2015年首次提出Attention Machanism的概念，原文中叫做Alignment。其思想就是在Decoder解码的过程中，为每一个时刻都计算一个Context Vec。秉着跟论文保持一致的原则，首先对需要用到的变量说明一下。假设Seq2Seq使用**LSTM**作为RNN单元，那么我们把Encoder每一时刻的输出记作$h_{t}$，而Decoder每一时刻的输出记作$s_{t}$，易得$h$的初始状态是零初始化的，而$h$的最后一个状态会被作为$s$的初始状态。

在Encoder完成编码之后，我们会得到Encoder所有时刻的输出：

$$
\vec{h}=\lbrack{h_{1},h_{2},\cdots,h_{T}}\rbrack
$$

其中$T$表示输入序列$X$的时间长度。然后$h_{T}$作为Decoder的初始状态$s_{0}$，开始解码并输出$s_{1}$。Align Model中每一次decode都需要考虑当前时刻与$X$序列每一个时刻的关系。令Decoder的第$1$时刻与$\vec{h}$各时刻的关系用一个分数向量$\vec{e}_{1}$来表示：

$$
\begin{aligned}
    \vec{e}_{1}&=\lbrack{e_{1,1},e_{1,2},\cdots,e_{1,T}}\rbrack \\
    e_{1,i}&=f(s_{0},h_{i}) \\
    &=w_{e}\cdot{tanh(w_{s}\cdot{s_{0}}+w_{h}\cdot{h_{i}})} \quad i\in\lbrack{1,T}\rbrack
\end{aligned}
$$

对$\vec{e}_{1}$做softmax归一化就可以得到一个和为$1$的权重向量$\vec{\alpha}_{1}$：

$$
\begin{aligned}
    \vec{\alpha}_{1}&=\lbrack{\alpha_{1,1},\alpha_{1,2},\cdots,\alpha_{1,T}}\rbrack \\
    \vec{\alpha}_{1}&=\frac{\exp(\vec{e}_{1})}{\sum_{T}\exp(\vec{e}_{1})} \\
\end{aligned}

$$

再将Encoder每个时刻的输出与对应位置的权重相乘再求和，就得到了Decoder该时刻的contex vec：

$$
\begin{aligned}
    c_{1}&=\sum\limits_{T}\vec{h}\cdot\vec{\alpha}_{1} \\
    &=\sum\limits_{T}h_{i}\cdot\alpha_{1,i} \\
\end{aligned}
$$

综上，Attention的运算过程为：

$$
\begin{aligned}
    \vec{e}_{t}&=W_{e}^{T}{\tanh(W_{s}{s_{t-1}}+W_{h}{\vec{h}})} \\
    \vec{\alpha}_{t}&=\frac{\exp(\vec{e}_{t})}{\sum_{T}\exp(\vec{e}_{t})} \\
    c_{t}&=\left< \vec{\alpha}_{t},\vec{h} \right> \\
\end{aligned}
$$

## Decoder

在此顺带一提运行Decoder的几种策略。

### Classic

一种是像[下图](https://satopirka.com/2018/02/encoder-decoder%E3%83%A2%E3%83%87%E3%83%AB%E3%81%A8teacher-forcingscheduled-samplingprofessor-forcing/)所示，Decoder每一时刻的输入都完全来自于上一时刻的输出。这种方式其实就是经典的RNN训练策略，它有一个明显的缺点，当Decoder某一时刻预测错误时，那么后面时刻的cell只会错的更加离谱。“差之毫厘谬以千里”是对该种方法最好的概括。

![](/img/without-teacher-forcing.png)

### Teacher Forcing

另一种方式就是使用偏移一个单位的真实标签(shifted target)作为输入，如[下图](https://satopirka.com/2018/02/encoder-decoder%E3%83%A2%E3%83%87%E3%83%AB%E3%81%A8teacher-forcingscheduled-samplingprofessor-forcing/)所示。这种方法叫做Teacher Forcing，它增强了Decoder训练时的稳定性，能加速模型的收敛。但是该方法无法用于测试(验证)，因为测试集的label是不知道的，所以测试模型时还需要换回Classic模式。

![](/img/teacher-forcing.png)

### BeamSearch

因为Teacher Forcing无法用于模型测试，那么考虑下Classic模式的测试过程。Decoder每个时刻的输出实际上经过了一个```argmax```运算，即只输出词库中概率最大的那个词。该方法的缺点之前已经提了，就是不稳定，某一个时刻错了后面就会继续错下去。

![](/img/2018101114371929.png)

BeamSearch方法改进了这一缺点，使用BeamSearch策略的Decoder每时刻的输出不再局限于单条路线，而是将搜索空间扩大成多条支线。假设BeamSearch的空间参数为$K$，那么Decoder每个时刻都会有$K$个输出，如下图所示。下图是使用BeamSearch策略的Decoder的搜索空间，其中$K=5$。下一时刻的搜索只会选取当前空间的TOP$K$个备选出发，概率不在TOP$K$的就会被抛弃。

![](/img/A-partially-completed-beam-search-procedure-with-a-beam-width-of-5-for-an-example-input.png)

## Self-Attention

传统的Seq2Seq模型由于其结构上的缺陷，从而没法并行训练

Transformer
