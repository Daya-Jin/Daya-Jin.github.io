---
layout: post
title:  "Linux Command"
categories: Linux
tags: Linux
---

* content
{:toc}

```ls [OPT] [FILE]```：列出文件列表
- ```-a```，列出有文件
- ```-l```，使用清单格式
- ```-h```，搭配```-l```使用，可读形式

```cd [PATH]```：change directory，切换目录

```mkdir [OPT] DIRs```：make directory，创建目录
- ```-p```，如需必要，创建父目录

```rm [OPT] FILE```：remove，删除文件或目录
- ```-f```，强制删除，无视错误
- ```-r```，递归删除目录及其子目录

```which CMD```：定位命令的路径

```touch [OPT] [FILEs]```：更改文件时间戳，文件不存在时创建空文件，常用于创建文件

```cp [OPT] SOURCE DEST```，copy，复制操作
- ```-f```，强制复制
- ```-r```，递归复制目录及其子目录

```mv [OPT] SOURCE DEST```，move，移动操作
- ```-f```，覆盖

```less FILE```：查看文件内容，空格换页，上下换行

```head [OPT] FILE```：查看文件的前$10$行
- ```-n```，指定查看前$n$行

```tail [OPT] FILE```：查看文件的前$10$行
- ```-f```，当文件增长时，输出追加内容
- ```-n```，指定查看前$n$行