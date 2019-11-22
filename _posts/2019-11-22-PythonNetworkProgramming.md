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