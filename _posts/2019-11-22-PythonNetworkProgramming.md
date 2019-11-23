---
layout: post
title:  "Python Network Programming"
categories: Network
tags: Python Network
---

* content
{:toc}

## Basic

一个简单的TCP回显服务器与客户端程序。

```server.py```：

```python
import socket

HOST = socket.gethostname()
PORT = 50007

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)    # 积压连接数
    conn, addr = s.accept()    # 返回新sock对象与连接地址
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(data)    # 发送所有数据
```

```client.py```：

```python
import socket

HOST = '192.168.10.128'  # The remote host
PORT = 50007  # The same port as used by the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b'Hello, world')    # 发送所有数据(字节流)
    data = s.recv(1024)    # 字节流
    print(type(data))
print('Received', repr(data))    # 字节流转字串
```

## 黏包

因为TCP是面向连接的协议，数据流是以packet的形式发送与接收的，如果发送消息过长会被分成多个包，同样接收方也会分多次接收。问题的根源在于，接收方不知道发送方要发送多长的数据，若发送方连续连续发送两条数据，接收方不知道数据之间的分割，因此将两条本应分开数据(的某部分)一起接收了，于是产生了黏包现象。UDP没有黏包问题。

黏包问题的解决方式也很简单，既然问题的根源在于接收方不知道发送数据的长度，那么在发送正式数据前发送方通知接收方的数据长度即可。

## Struct

Python的```struct```模块是一个字节流解释器跟翻译器，一个最简单的示例，现有三个数1、2、3，如果要将其分别按不同的类型转成(翻译)字节流，可以使用下述代码：

```python
import struct

print(struct.pack('ihb', 1, 2, 3))    # b'\x01\x00\x00\x00\x02\x00\x03'
print(struct.pack('!ihb', 1, 2, 3))    # b'\x00\x00\x00\x01\x00\x02\x03'
```

这里解释一下程序与输出。```pack```的第一个参数是格式字串，其后跟的全是需要转换的数据，格式参数必须跟数据一一对应。上述程序中的```!```表示网络字节序，```i```表示int(4Byte)，```h```表示short(2Byte),```b```表示signed char(1Byte)。根据```ihb```的解释格式，1、2、3会被分别转换成：```0x00000001```、```0x0002```和```0x03```，然后按照不同的字节序会得到不同的字节流。

又比如，在ping程序的ICMP报文中，前$8$个字节即为ICMP的头部，如下图所示，ICMP的头部字节码为：```\x08\x00\x4c\x60\x00\x01\x00\xfb```。

![](/img/2019-11-22_22-28-33.jpg)

这里对ICMP的首部进行解码，由人工计算易得首部各字段的值应该为```(8, 0, 19552, 1, 251)```，由```struct```模块解码得到的值与之一致：

```python
import struct

data = b'\x08\x00\x4c\x60\x00\x01\x00\xfb'
print(struct.unpack('!BBHHH', data))    # (8, 0, 19552, 1, 251)
```

