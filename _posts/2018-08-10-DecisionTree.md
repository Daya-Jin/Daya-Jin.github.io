---
layout: post
title:  "Decision Tree"
categories: MachineLearning
tags: tree
---

* content
{:toc}

## 模型概述

决策树算法最早用于分类任务，算法根据数据的特征与类别生成一棵树，并以这棵树对未知数据进行分类。

首先要了解熵(Entropy)的概念。在热力学中，熵被用于表示系统的混乱程度；而在信息论中，熵用于表示信息量的大小。

在一个有$K$个类别的样本$D$中，假设类别$Y$的概率分布为：

$$
P(Y=k)=p_{k}
$$

那么这个具有$K$个类别的样本的信息熵为：

$$
H(D)=-\sum_{k=1}^{K}p_{k}\log{p_{k}}
$$

当整个数据集只有一个类别时，熵最低，为

$$
H_{min}(D)=-1\cdot{log1}=0
$$

当数据集各类别为均匀分布时，熵最大，为

$$
H_{max}(D)=-K\cdot\frac{1}{K}\cdot\log\frac{1}{K}=-\log{K}
$$

假设某一离散属性$A$有$V$个不同的的取值，那么当以特征$A$来划分时可以将数据集$D$分为$V$个子集：

$$
D=\sum_{v}^{V}D_{v}
$$

那么划分之后的$V$个子数据集的加权熵为：

$$
H(D|A)=\sum_{v}^{V}\frac{|D_{v}|}{|D|}H(D_{v})
$$

## ID3

Iternative Dichotomizer，是最早的决策树算法，其根据**信息增益**(Information Gain)来寻找最佳决策特征，当按特征A来划分数据集时的信息增益定义为：

$$
G(D,A)=H(D)-H(D|A)
$$

首先，树的根节点中包含整个数据集，ID3算法会遍历所有特征分别做**test**来计算划分后的信息增益，然后选择信息增益最大的那个test，按该特征的类别数将数据集划分到下一层的各个子节点中；对各个子节点中的数据递归进行这个过程，直到信息增益足够小或者无特征可用。

可以看出，ID3算法有如下特点：

- 贪心算法
- 每次做决策时只用一个特征
- 没有办法处理连续特征跟缺失值
- 容易过拟合

在只考虑信息增益的情况下，ID3算法有一个致命缺陷，就是会倾向于选择类别数多的特征来做划分。假设有一列特征(如样本ID)类别数与样本数相等，如果以该特征来进行划分数据集，则数据集被划分成了单样本节点，每个节点的熵均为0，总熵也为0，这样一来得到了最大的信息增益，但是这种划分显然是不合理的。

实现指导：[分类](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/ID3_Clf.ipynb) &ensp; [回归](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/ID3_Reg.ipynb)

完整代码：[分类](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/ID3_Clf.py) &ensp; [回归](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/ID3_Reg.py)

## C4.5

为了修正ID3算法的缺陷，C4.5算法应运而生。

首先，C4.5在ID3的基础上改进了生成树算法，不再使用信息增益，而是使用**增益比**(gain ratio)来决定使用哪个特征来划分数据集。

增益比定义为：

$$
GR(D,A)=\frac{G(D,A)}{IV(A)}
$$

其中$IV(A)$被称为**固有值**(intrinsic value)，它等价于某一特征在数据集中的熵。

假设在一个数据集$D$中有某个特征$A$，其有$K$个不同的取值，那么特征$A$在数据集中的概率分布为：

$$
P(A=v)=\frac{|D_{v}|}{|D|}=p_{v}
$$

那么数据集中该特征的固有值(熵)为：

$$
IV(A)=-\sum_{v=1}^{V}p_{v}\log{p_{v}}
$$

这样一来就减少了信息增益在做决策时的比重。

此外，C4.5算法还加入了对连续值的处理。对于一个连续特征$A$，假设其在样本中有$n$个不同的取值，其有序集合为$\{a_{1},a_{2},…,a_{n}\}$，那么对应的就有$n-1$处划分点，取相邻两值的中点做划分点，那么划分点集合可表示为：$s_{i}=\{\frac{a_{i}+a_{i+1}}{2}\vert{i=1,2,...,n-1}\}$。当选取某一划分点进行划分时，数据集$D$被分成两部分：$D^{-}$中所有样本的$A$特征都要小于等于$s_{i}$，$D^{+}$中所有样本的$A$特征都要大于$s_{i}$。然后再以之前的公式计算划分之后的信息增益与增益比。

然后是C4.5对缺失值的处理，决策树算法在处理缺失值熵需要解决三个问题：熵的计算，数据集的划分，带缺失值的预测。

第一条，在做test时，当前特征为缺失值的样本不参与熵的计算；

第二条，在做test时，给当前特征缺失的样本赋一个初始值为$1$的权重，做划分时，这些特征缺失的样本会被复制多份分配到所有子节点中，并按照叶子节点中非缺失样本的比例更新权重。设某个子结点中的非缺失样本占父节点非缺失样本比例为$r$ ，则更新该结点中每个缺失值样本的权重为$w:=w\cdot{r}$。

第三条，在预测带缺失值的样本时，样本在决策树中的行走路线同上，最后该样本会同时出现在多个叶子节点中，按在各叶子节点中的权重和来给出预测类别。

最后，C4.5使用自底向上的剪枝策略来避免过拟合。

## Classification And Regression Tree

CART是一棵二叉树，每次test都将问题空间分成两个区域，可处理连续变量与类别型变量，所以one-hot encoding对CART是无效的。

**分类**

CART分类树的生成类似上面两种算法，不过做test时选取特征的依据是**基尼指数**(Gini Index)。对于有$$K$$个类别的数据集$$D$$，某一样本属于类别$$k$$的概率等于该类别的分布概率：

$$
P(Y=k)=p_{k}
$$

那么该数据集$$D$$的基尼指数定义如下：

$$
Gini(D)=1-\sum_{k=1}^{K}p_{k}^{2}
$$

从公式易得，基尼指数表示的是从数据集中随机取两个不同类别样本的概率，其值越小则数据集纯度越高。

由于CART的特殊性，其在做test时与ID3、C4.5略有不同，CART是二叉树，并且在对**类别型**变量做划分时做得是非划分。如对一个数据集$$D$$以特征$$A$$的一个取值$$a$$来划分，那么数据集会被划分成$$D_{A=a}$$和$$D_{A\ne{a}}$$，那么数据集$$D$$依照特征$A=a$划分之后的加权基尼指数为：

$$
G(D|A=a)=\frac{|D^{A=a}|}{|D|}Gini(D^{A=a})+\frac{|D^{A{\ne}a}|}{|D|}Gini(D^{A{\ne}a})
$$

CART在划分时会遍历所有的特征与其所有可能的取值，再全局考量选取一个最佳特征与最佳划分点。若数据集有$M$个特征，每个特征有$V$种不同取值，上述过程可以用下式来表述：

$$
(A_{opt},a_{opt})=arg\ min(G(D\vert{A_{i}}=a_{j})) \qquad i\ from\ 1 \to M,j\ from\ 1\to V
$$

[实现指导](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/DecisionTreeClassifier.ipynb)

[完整代码](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/DecisionTreeClassifier.py)

**回归**

首先需要说明的是回归树的输出是叶子结点中所有样本目标值的均值。那么对于回归任务，怎么去生成树？可以采用一种直观的方式来对数据集进行划分，假设某一时刻数据集(数据子集)$D$被决策树以特征$X_{j}$按取值$s$划分成了两部分(或两个叶子节点)：

$$
R_{1}(X_{j},s)=\{X\vert{X_{j}<s}\} \\
R_{2}(X_{j},s)=\{X\vert{X_{j}\ge s}\} \\
$$

则此次划分的优劣可用MSE来判断：

$$
Loss(X_{j},s)=\sum_{R_{1}}(y_{i}-\bar{y}_{R_{1}})^{2}+\sum_{R_{2}}(y_{i}-\bar{y}_{R_{2}})^{2}
$$

其中，$\hat{y}_{R_{1}}$为$R_{1}$区域所有样本目标值的均值，$\hat{y}_{R_{2}}$为$R_{2}$区域所有样本目标值的均值。

在生成回归树时，使用贪心策略，遍历所有特征下所有可能的取值，找到一个最优划分点，然后以此类推。决策树在做预测时，首先将测试样本丢进决策树进行判定，判定该测试样本属于哪一个叶子节点，然后把该叶子结点内所有训练样本的目标均值作为测试样本的预测值。

除了使用MSE作为**分裂依据**(splitting criteria)之外，还有一个变量可以用作回归树的分裂：**方差**(variance)。当使用方差作为分裂依据时，生成树的目的很明显，希望子节点内的值越稳定越好。

CART同样使用剪枝来避免过拟合。

**注意：**能做回归任务并不是CART树的专利，ID3与C4.5都可以用于回归！

[实现指导](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/DecisionTreeRegressor.ipynb)

[完整代码](https://github.com/Daya-Jin/ML_for_learner/blob/master/tree/DecisionTreeRegressor.py)

## 树的正则化

决策树算法的缺点就在于极易过拟合，所以控制决策树的模型复杂度以防止过拟合是很有必要的。首先可以设定几个参数抑制树的生长：**最大树深**(max_depth)，**最大叶子结点数**(max_leaf_nodes)，**叶子结点最小样本数**(min_samples_leaf)，**分裂最小增益**(min_impurity_decrease)。除此之外，也可以对树的生长不做限制，然后再对树进行**剪枝**(pruning)。

### Pessimistic Estimate Pruning

C4.5使用悲观估计剪枝法(待补充)。

### Cost-Complexity Pruning

CART使用cost-complexity剪枝方法。类似线性模型中的正则化，可以加入一项结构风险项来指导剪枝过程，定义一个**Cost-Complexity**函数：

$$
C_{\alpha}(T)=Err(T)+{\alpha}L(T)
$$

其中$T$表示一棵树，$Err(T)$表示树$T$的分类(回归)误差，$\alpha$为正则化系数，$L(T)$为能表示树结构复杂度的函数。下面以分类任务做示例。

令树$T$的结构复杂度函数等于树的叶节点数：

$$
L(T)=\vert{T}\vert
$$

再令树$T$的误差函数为：

$$
Err(T)=\sum_{i=1}^{|T|}err(t_{i})p(t_{i})
$$

其中$t_{i}$为树$T$中的第$i$个叶节点，$err(t_{i})$为该叶节点的分类误差率，$p(t_{i})$为该叶节点的样本比例，上式实质上是一个加权误差率。对一个确定的$\alpha$值，一定会有一颗最小化$C_{\alpha}$的树$T_{\alpha}$。为了找到这颗最优剪枝树$T_{\alpha}$，使用**最弱链接剪枝**(weakest link pruning)策略，自底向上地对非叶节点进行剪枝并查看效果，然后选取一个表现最好的$T_{\alpha}$。

假设某一时刻以节点$t$进行剪枝，那么剪枝后与剪枝前的CC函数差为：

$$
\begin{aligned}
\Delta C_{\alpha}(t)&=C_{\alpha}(T-T_{t})-C_{\alpha}(T) \\
&=Err(T-T_{t})-Err(T)+\alpha(\vert{T-T_{t}}\vert-\vert{T}\vert) \\
&=(-Err(T_{t})+err(t))+\alpha(-\vert{T_{t}}\vert+1) \\
&=err(t)-Err(T_{t})+\alpha(1-\vert{T_{t}}\vert) \\
\end{aligned}
$$

其中，$T_{t}$为树$T$中以节点$t$为根节点的子树。令$\Delta C_{\alpha}(t)=0$得$g(t)=\alpha'=\frac{err(t)-Err(T_{t})}{\vert{T_{t}-1}\vert}$，整个CCP算法流程如下所述：

1. 生成一颗完整树$T^{0}$，对所有的非叶节点都进行剪枝尝试，找到一个最小化$g(t_{1})$的剪枝节点$t_{1}$，令$\alpha^{1}=g(t_{1})$，$T^{1}=T^{0}-T_{t_{1}}$

2. 对$T^{1}$所有的非节点都进行剪枝尝试，找到一个最小化$g(t_{2})$的剪枝节点$t_{2}$，令$\alpha^{2}=g(t_{2})$，$T^{2}=T^{1}-T_{t_{2}}$

3. 依次进行下去，直到只剩下一个根节点为止，那么可以得到一个子树序列$[T^{0},T^{1},...,root]$和一系列参数$[\alpha^{1},\alpha^{2},...]$，然后在所有子树上使用交叉验证来选取一个最佳参数$\hat{\alpha}$与最佳剪枝树$T_{\hat{\alpha}}$。

## 总结

决策树算法的优缺点分别在于

优点：

- 可解释性强，甚至强于线性模型
- 可可视化
- 不需要太多的数据预处理
- 能同时处理数值型数据与类别型数据
- 自带多分类解决方案

缺点：

- 预测准确性不如其他模型，因为树模型中没有统计模型
- 树模型的variance很大，一方面是模型本身极易过拟合，非常不稳定；另一方面是顶层划分的误差会逐级往下传播
- 树模型的if-else规则难以捕获到数据中的非轴向关系
- 做回归任务时缺少平滑性，需要增加树深来提高表现

各决策树算法的对比：

|          | ID3  | C4.5 | CART |
| :--------: |: ----: | :--: | :--: |
| splitting criterion | Information Gain | gain ratio | Gini Index |
| tree structure | multiway tree | multiway tree | binary tree |
| continuous attribute support | - | √ | √ |
| handling missing value | - | √ | √ |
| pruning | - | √ | √ |
| regression support | - | - | √ |

# 集成学习

讲决策树就不得不提集成学习。因为树模型中没有引入统计模型，导致树模型对的variance很大，虽然在训练集上表现很好，但是在测试集上的表现却差强人意，所以集成学习很适合应用到决策树算法上。
