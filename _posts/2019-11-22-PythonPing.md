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

![](/img/2019-11-22_22-28-33.jpg)

上图是在Windows平台下ping某网站时的wireshark抓包，其中高亮部分为ICMP报文，易得ICMP的头部(前$8$Byte)值为：

```
08 00 4c 60 00 01 00 fb
```

后面的值为报文装载数据，可以看出是从a到i的顺序字母串。下面以此为例，使用Python实现ICMP的校验和计算。

首先是报文的构建，这里以发送方为例。根据wireshark抓包，易得首部的所有字段情况，其中Checksum字段需要置零：

```python
type = 8  # Type: '\x08'(ICMP Echo Request)
code = 0  # Code: '\x00'
checksum = 0  # Checksum
id = 1  # ID: '\x00\x01'
seq = 251  # Sequence: '\x00\xfb'
body = b"abcdefghijklmnopqrstuvwabcdefghi"  # Data
icmp_msg = struct.pack('!BBHHH32s', type, code, checksum, id, seq, body)
```

然后就是分组、累加，因为此处报文是根据网络字节序拼接形成的，所以在做分组加法时，后面的值为高位，前面的值为低位：

```python
acc = 0
for i in range(0, len(icmp_msg), 2):  # 16bit一组
    group_val = icmp_msg[i] + (icmp_msg[i + 1] << 8)  # 16bit的值，注意字节顺序
    acc += group_val
    acc = (acc & 0xffff) + (acc >> 16)
```

最后一步是取反，还要将字节序转换成网络字节序，最后得到的网络字节序校验和```n```应该和上图抓包中的保持一致：$\x4c60$，即十进制的$19552$：

```python
h = ~acc & 0xffff  # host byte order(取反并截取低16位)
n = h >> 8 | (h << 8 & 0xff00)  # network byte order(高8位低8位互换)
assert n == 19552  # '\x4c60'
```

封装的ICMP校验和代码[见此](https://github.com/Daya-Jin/As_a_Programmer/blob/master/Python/ICMP_checksum.py)。

## Ping

