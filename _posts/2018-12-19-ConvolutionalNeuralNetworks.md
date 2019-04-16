---
layout: post
title:  "Convolutional Neural Networks"
categories: DeepLearning
tags: DeepLearning CNN
---

* content
{:toc}

# Convolutional Neural Networks

## 概述

在普通的深度神经网络(DNN)中，每个单元都可看做是一个神经元，每一层的神经元都会接受来自上一层所有神经元的信号；句话说，前一层即使只有一个神经元是兴奋的，它也会激活后面所有层的所有神经元。这种人工神经网络设计并不符合生物神经学，生物学家发现动物在接收不同的刺激时，大脑中活跃的区域是不一样的，这就说明神经元之间并不是**全连接**(Fully Connected)关系，而是一种选择性连接的关系。这就是**卷积神经网络**(Convolutional Neural Networks)诞生的起源。

CNN最初也是最广泛应用的领域就是图像处理，假设有一张$32{\times}32$的RGB图片，再假设眼球的神经元能看到的范围为$5{\times}5$，并且能接受RGB信号，那么会有类似如下的结构：

![](/img/2019-04-14_11-03-11.bmp)

$32{\times}32{\times}3$表示的是输入卷积网络的图片，$5{\times}5{\times}3$的filter也被称为卷积网络的卷积核，用于提取图片的区域特征。CNN使用卷积核去扫描图片，每扫描一个区域会得到一个输出信号，那么该图片经过卷积核的扫描之后会得到一个信号矩阵：

![](/img/2019-04-14_11-18-54.bmp)

具体过程见下图(图源百度)：

![](/img/a0263addeb2e19b74cbbddedb6abc71e.gif)

信号矩阵在经过激活函数激活之后会得到一个激活矩阵，这一处理过程被称为**激活映射**(Active Mapping)。使用一个卷积核会得到一个激活矩阵，那么使用多个卷积核会得到多个激活矩阵，即该层网络的输出，也是下层网络的输入：

![](/img/2019-04-14_11-29-37.bmp)

在CNN中，执行激活映射的层被称为**卷积层**(Convolution Layer)。设输入尺寸为$nn$，卷积核的尺寸为$k$，那么卷积层的输出为$n-k+1$。卷积核还有一个可设定的参数，**步长**(stride)，该参数指明卷积核每次扫描移动的行列数，在该参数设定的条件下，卷积核的输出尺寸为$(n-k)/s+1$。经过若干卷积层之后，网络中数据的尺寸是越来越小的：

![](/img/2019-04-14_13-14-22.bmp)

为了防止数据尺寸缩小的太快，CNN会使用一种padding的技术，对数据的边缘填充0值来增大输入数据的尺寸，那么在使用padding技术时的输出尺寸为：

$$
o=(n+2p-k)/s+1
$$

除了卷积层，CNN中还有**池化层**(Pooling Layer)。池化层的目的很简单，就是对数据做**降采样**(downsampling)。上面提到不使用padding的卷积层会使数据尺寸变小，对应的，在池化层中，不使用padding，并且设置较大的stride来达到数据缩减的目的。最常见的池化方法有平均池化与最大化池化。下图为最大化池化的示例(图源百度)：

![](/img/f0706474f3855782b7dca7c06cfbcbb5.gif)

一个完整的CNN包含若干个卷积层与池化层，最后是全连接层。**注意**只有卷积层才有激活函数。

## 训练

CNN中的卷积核相当于DNN中的权重矩阵，那么CNN中的参数即是卷积核张量。一个尺寸为$k{\times}k{\times}d$的卷积核，其中的参数数量为：

$$
k{\times}k{\times}d+1
$$

其中$+1$表示的是bias。

## CNN Architectures

### AlexNet

AlexNet算是首个成功的CNN结构，其在2012年的ImageNet图像分类比赛中获得冠军。其网络结构如下所示：

![](/img/2019-04-15_19-26-30.bmp)

||Input|kernel num|kernel size|stride|pad|Output|
|-|:-:|:-:|:-:|:-:|:-:|:-:|
|Conv1|$227{\times}227{\times}3$|$96$|$11{\times}11$|4|0|$(227-11)/4+1$|
|MaxPool1|$55{\times}55{\times}96$|-|$3{\times}3$|2|-|$(55-3)/2+1$|
|Norm1|$27{\times}27{\times}96$|-|-|-|-|27|
|Conv2|$27{\times}27{\times}96$|$256$|$5{\times}5$|1|2|$(27+2{\times}2-5)/1+1$|
|MaxPool2|$27{\times}27{\times}256$|-|$3{\times}3$|2|-|$(27-3)/2+1$|
|Norm2|$13{\times}13{\times}256$|-|-|-|-|27|
|Conv3|$13{\times}13{\times}256$|$384$|$3{\times}3$|1|1|$(13+2{\times}1-3)/1+1$|
|Conv4|$13{\times}13{\times}384$|$384$|$3{\times}3$|1|1|$(13+2{\times}1-3)/1+1$|
|Conv5|$13{\times}13{\times}384$|$256$|$3{\times}3$|1|1|$(13+2{\times}1-3)/1+1$|
|MaxPool3|$13{\times}13{\times}256$|-|$3{\times}3$|2|-|$(13-3)/2+1$|
|FC6|$6{\times}6{\times}256$|4096|-|-|-|$1{\times}4096$|
|FC7|$1{\times}4096$|4096|-|-|-|$1{\times}4096$|
|FC8|$1{\times}4096$|1000|-|-|-|$1{\times}1000$|

AlexNet有几个关键点：
- 首次在实践中使用ReLU作为激活函数
- 使用了归一化层来做局部归一化(LRN)
- 使用dropout技术避免过拟合
- 做了大量的数据增强(图像翻转、旋转)
- 全部使用最大池化，并且池化核的步长小于核大小，使池化核之间有重叠
- 使用两个GPU并行训练
- 使用带动量的SGD，当损失不再下降时手动将学习率除以10

### VGG

VGG是2014年ImageNet图像分类挑战的亚军，其扩展了AlexNet的结构，使用了更深层的神经网络模型来达到更好的效果。下图是VGG16的网络结构图：

![](/img/vgg_pa03.jpg)

VGG中只使用了两种核：

||kernel size|stride|padding|
|-|:-:|:-:|:-:|
|Conv Kernel|$3{\times}3$|$1$|$1$|
|MaxPooling Kernel|$2{\times}2$|$2$|-|

易得VGG中的卷积层不改变数据的尺寸，但是会增加数据的深度；而最大池化层每次都会将数据的尺寸减半，但深度不变。这样一来，数据在VGG中传递时尺寸不断减小，深度不断增加，自然地过渡到一个一维向量(预测输出)。

并且级联的小核卷积层相当于一个单个的大核卷积层，但是参数数量却大大降低。假设输入图片维度为$(7,7,3)$，在经过两个使用$3{\times}3{\times}3$卷积核的级联卷积层后，数据流维度变为$(3,3,3)$，而两层六个卷积核的参数数量为：$6{\times}3{\times}3{\times}3=162$；假设把两层小核卷积层换成一个使用$5{\times}5{\times}3$卷积核的卷积层，输出数据流维度同样为$(3,3,3)$，那么三个大卷积核的参数数量为：$3{\times}5{\times}5{\times}3=225$。除了减小参数数量之外，增加的卷积层为神经网络增加了更多的非线性因素。

接下来分析VGG16各层需要的内存量(数据流占用内存)与参数数量：

|layer|mem|param|
|-|:-:|:-:|
|Input|$224{\times}224{\times}3=150K$|0|
|Conv1-1|$224{\times}224{\times}64=3.2M$|$64{\times}3{\times}3{\times}3$|
|Conv1-2|$224{\times}224{\times}64=3.2M$|$64{\times}3{\times}3{\times}64$|
|MaxPool1|$112{\times}112{\times}64=800K$|0|
|Conv2-1|$112{\times}112{\times}128=1.6M$|$128{\times}3{\times}3{\times}64$|
|Conv2-2|$112{\times}112{\times}128=1.6M$|$128{\times}3{\times}3{\times}128$|
|MaxPool2|$56{\times}56{\times}128=400K$|0|
|Conv3-1|$56{\times}56{\times}256=800K$|$256{\times}3{\times}3{\times}128$|
|Conv3-2|$56{\times}56{\times}256=800K$|$256{\times}3{\times}3{\times}256$|
|Conv3-3|$56{\times}56{\times}256=800K$|$256{\times}3{\times}3{\times}256$|
|MaxPool3|$28{\times}28{\times}256=200K$|0|
|Conv4-1|$28{\times}28{\times}512=400K$|$512{\times}3{\times}3{\times}256$|
|Conv4-2|$28{\times}28{\times}512=400K$|$512{\times}3{\times}3{\times}512$|
|Conv4-3|$28{\times}28{\times}512=400K$|$512{\times}3{\times}3{\times}512$|
|MaxPool4|$14{\times}14{\times}512=100K$|0|
|Conv5-1|$14{\times}14{\times}512=100K$|$512{\times}3{\times}3{\times}512$|
|Conv5-2|$14{\times}14{\times}512=100K$|$512{\times}3{\times}3{\times}512$|
|Conv5-3|$14{\times}14{\times}512=100K$|$512{\times}3{\times}3{\times}512$|
|MaxPool5|$7{\times}7{\times}512=25K$|0|
|FC6|$1{\times}4096=4K$|$7{\times}7{\times}512{\times}4096=98M$|
|FC7|$1{\times}4096=4K$|$4096{\times}4096=16M$|
|FC8|$1{\times}1000=1K$|$4096{\times}1000=4M$|

根据以上表格，发现计算时的最大内存开销在于前面两层卷积层，而参数的数量绝大部分都在全连接层。以每一个数字占$32$bit来算，仅计算一张图片在VGG中的前向传播过程就至少需要$60$MB的内存，保存VGG模型的所有参数至少需要$526$MB的存储空间。

### GoogLeNet

GoogLeNet是2014年ImageNet图像分类挑战的冠军，它同样扩展了AlexNet的网络深度，但不同于VGG，GoogLeNet使用了一种全新的网络子结构来减少参数与运算量。

**Interception Module**

Interception Module是GoogLeNet中出现的全新网络子结构，如下图所示：

![](/img/nception-module-of-GoogLeNet-This-figure-is-from-the-original-paper-10.png)

一个Interception Module对上一层的输出并行地做了三次基于不同卷积核的卷积运算与一次池化运算，并且引入了$1{\times}1$的卷积核来减少数据流的深度，这样就降低了参数数量与运算量。

Interception Module最终会将四个运算结果沿深度轴拼接起来，那么四个运算结果的数据流深度可以不同，但是尺寸必须相同。一个Interception Module中可能的数据流尺寸如下图所示：

![](/img/2019-04-16_09-38-31.bmp)

注意到$1{\times}1$卷积核所在层的数据流深度，比其上一层与其下一层的数据流都要小，所以$1{\times}1$卷积核所在的层也叫做“**瓶颈层**”(bottleneck layer)。

完整的GoogLeNet结构如下图所示：

![](/img/1_ZFPOSAted10TPd3hBQU8iQ.png)

关于GoogLeNet，值得一提的有几点：

- 在最后一个Module之后有一个AveragePool层，其目的是为了替代传统的FC层，大大减少参数量，其后添加的FC层只是为了便于调优
- 在网络中间层添加了两个额外的中间输出，目的是为了避免梯度消失；同时中间输出会以一定的权重加到最终输出上去

### ResNet

有人经过实际测试发现，单纯的将网络加深并不能提升网络的表现，深层网络的表现反而还不如浅层网络，甚至深层网络在训练集上的误差都要高于浅层网络。这个结论是反直觉的，因为如果深层网络是容易过拟合的话，那么至少在训练集上的表现不会弱于浅层网络，可能的解释就是深层网络的模型空间太大导致太难训练。假设在深层网络的后面几层直接将输入输出，即该层什么都不做，那么就可以得到一个与浅层网络等价的深层网络。受此启发，ResNet诞生，ResNet中的**残差块**(residual block)如下图所示：

![](/img/Bottleneck-Blocks-for-ResNet-50-left-identity-shortcut-right-projection-shortcut.png)

每一个残差块包含两层使用$3{\times}3$卷积核的卷积层，除此之外，残差块在激活输出之前，会将输入加到输出上，相加时会对输入再做一次卷积保证输入输出的深度相等。之前的GoogLeNet使用了$1{\times}1$的卷积核来减少参数数量与运算量，对于过深的ResNet，同样使用瓶颈层来提高模型的训练效率。完整的ResNet结构就是若干个残差块的级联，并且ResNet的性能在深度过深时不会受到影响。

![](/img/1_2ns4ota94je5gSVjrpFq3A.png)

注意ResNet在最后同样没有FC层，大大减少了参数数量，并且使用了与GoogLeNet相同的平均池化。