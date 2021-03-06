---
layout: post
title:  "Operating System"
categories: system
tags: OS
---

* content
{:toc}

# 概述

复习笔记，应付秋招。

## Process & Thread

早期的**进程**(Process)是资源分配的最小单位，进程拥有自己的内存空间、文教描述符与上下文等资源，进程内的一个或多个**线程**(Thread)则共享进程资源。

进程的当前状态都被记录在一个**程序控制块**(Process Control Block)中，对单CPU而言，在任意时刻只有一个PCB处于激活状态，所有进程的PCB都登记在**进程表**(Process Table)中，OS会根据调度算法来调度不同进程的运行，进程之间的切换机制称为**中断**(Interrupt)。进程的状态转换如下图所示：

![](/img/states_modified.png)

一般认为进程有以下几种状态：

- **就绪**(Ready)，进程已具备**运行**的条件，但是还没有被调度；
- **运行**(Running)，调度算法选取一个**就绪**的进程运行；
- **阻塞**(Blocked)，进程正常**运行**所需的条件未被满足，如资源未请求成功或者等待某个事件发生，进入**阻塞**状态；
- **挂起**(Suspended)，因内存不足，未正常**运行**的进程会被移动到外存，进入**挂起**状态。

因为进程的执行与资源分配都是互斥的，所以进程可以看做是OS中的重型单位。进程之间的切换(中断)比较耗资源，由此提出**线程**(Thread)的概念，用于解耦执行与资源分配。线程是CPU调度的最小单位，同一进程中的所有线程都有相同的地址空间，因此线程之间是没有保护的。

为了保持线程之间的独立性，线程自己所拥有的东西为：PC、寄存器、堆栈和状态。因为线程切换所需要保存/恢复的变量比较少，所以切换线程的开销也要小于进程。

## IPC

由于进程拥有各自的地址空间及资源，那么**进程间通信**(Inter Process Communication)就必须通过一些别的方法来实现。

**管道**(pipe)用于进程间的数据传输，默认的匿名管道是半双工的，只能用于有亲缘关系的进程。而**有名管道**(Named Pipe)可以用于任意进程之间的通信。

**信号**(signal)用于进程间的消息传递与事件通知，信号是在软件层上对中断的一种模拟。信号来源有两个：硬件，如键盘鼠标；软件，如```kill```命令。

**消息队列**(Message Queue)通过为进程设置读写权限，允许任意两个进程间的通信。

**共享内存**(Shared Memory)指不同进程都可以访问的一块内存区域，不同进程均可以对该区域做读写操作。共享内存引入了一致性问题。

**信号量**(semaphore)实际上只是一个非负整数，指示可申请的资源数，用于进程间的同步操作。信号量机制中有两种操作：P(Proberen)和V(Verhogen)，分别表示尝试和增加。当某一进程需要申请(访问)资源时，首先需要检查信号量$S$。若$S==0$，说明当前无空闲资源，则进程阻；否则分配资源$S=S-1$，并且进程运行。进程运行完毕，执行V操作，释放资源$S=S+1$。PV操作对于信号量$S$的修改是原子操作。

**套接字**(socket)常用与网络中不同进程间的通信。

## Synchronization

在多程序系统中，**同步**主要指的是维持程序之间运行的协调性，重点在于保护数据的一致性。对共享内存进行访问的程序片段称为**临界区**(critical region)，很显然要保持数据的一致性就需要满足两个进程不能同时进入临界区。这一特性也叫做**互斥**(mutual exclusion)，即进程A与进程B在不能同时操作某部分数据。

临界区的互斥访问虽然保护了共享数据的一致性，但是也引入了另外一个问题：**死锁**(deadlock)。假设进程$A$、$B$都需要资源$1$和$2$才能正常运行，若某一时刻进程$A$占有了资源$1$，进程$B$占有了资源$2$，这种情况下$A$请求不到$2$，而$B$请求不到$1$，两者均会一直阻塞下去，这就是死锁。死锁的发生需要四个条件：

- 互斥，同一资源不能同时被多个进程所占有；
- 占有和等待，进程会一直占有已有资源，并会一直等待所需资源；
- 不可抢占，进程不可抢夺其他进程所占有的资源；
- 环路等待，在资源与进程的拓扑图中，一定存在环路需求关系。

要解决死锁问题有几种方法：

1. 检测和恢复，实时检测是否发生了死锁，若发生死锁则进行死锁恢复；
2. 死锁避免，在分配资源前预测是否可能发生死锁，若有死锁危险则不予分配，挂起进程；
3. 死锁破坏