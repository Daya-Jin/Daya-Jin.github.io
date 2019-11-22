---
layout: post
title:  "Python Ping"
categories: Network
tags: Python Network
---

* content
{:toc}

## ICMP

**网际控制协议**(Internet Control Message Protocol)运行在IP之上，属于TCP/IP协议簇的核心协议之一。其在网络中为数不多的直接使用场景就有ping命令。

ICMP报文头的形式如下图所示，共8Byte。在使用ping命令时，Type值固定为$08$，Code值固定为$00$，Checksum在由发送方计算并填充，ID与Sequence则不固定。

![](/img/2019-11-22_21-54-05.jpg)

需要注意的是ICMP的校验和是根据整个ICMP报文来计算的，发送方的计算过程如下所示：

1. 将Checksum字段置零，然后将整个首部按$16$bit为单位分组进行累加，若中间结果超出$16$bit(类似于进位)，则截取高位再加到低位上去；
2. 对累加和取反码，即为Checksum。

而接收方的校验则更简单，直接将ICMP报文按$16$bit分组，累计求和并取反，结果为$0$则确认，否则失败。

