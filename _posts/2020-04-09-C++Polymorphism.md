---
layout: post
title:  "C++ Polymorphism"
categories: programming
tags: C++
---

* content
{:toc}

## 多态

### 需求

从示例来讲C++中的多态。现在有如下需求：某游戏包含若干英雄，每个英雄有攻击动作和受伤反应。一个最蠢的实现就是定义一个英雄基类，然后再派生出若干子类，**在子类中**实现每个英雄的行为，需要注意的是，攻击动作会有一个目标英雄，并且攻击动作会引发目标英雄的受伤反应。那么一个最简易的实现代码如下所示。```Hero.h```：

```c++
#include<string>
using namespace std;

class Hero {};

class Yi;    // 为避免互相引用所必要的前置声明

class Garen :public Hero {
	const string name = "Garen";

public:
	void Attack(Yi* p);
	void Hurted();
};

class Yi :public Hero {
	const string name = "Yi";

public:
	void Attack(Garen* p);
	void Hurted();
};
```

```Hero.cpp```：

```c++
#include<iostream>
#include "Hero.h"
using namespace std;

void Garen::Attack(Yi* p) {
	cout << this->name << " attacks!" << endl;
	p->Hurted();
}

void Garen::Hurted() {
	cout << this->name << " was attacked!" << endl;
}

void Yi::Attack(Garen* p) {
	cout << this->name << " attacks!" << endl;
	p->Hurted();
}

void Yi::Hurted() {
	cout << this->name << " was attacked!" << endl;
}
```

```main.cpp```：

```c++
#include<iostream>
#include "Hero.h"

int main(void) {
	Garen garen;
	Yi yi;
	garen.Attack(&yi);
	yi.Attack(&garen);
	return 0;
}
```

上述代码构成的程序是正常运行的：

>Garen attacks!\
Yi was attacked!\
Yi attacks!\
Garen was attacked!

但是容易发现如果要新增一个英雄的话，比如现要新增一个名为"EZ"的新英雄，上述代码的改动是可预见的：所有旧英雄中都需要新增一个成员函数```void Attack(EZ* p)```，并且新英雄```EZ```类中需要完成对所有旧英雄的攻击代码。这样的改动量是无法接受的，注意到```Hero```基类所派生的所有子类都有```Attack```和```Hurted```行为，只是应用对象的类不同，自然而然想到如下思路：在基类中实现，由子类继承，并在调用时重载。

### 问题

C++自带基类与派生类的互相转化机制，因此更改后的代码如下所示。```Hero.h```：

```c++
#include<string>
using namespace std;

class Hero {
	const string name = "Hero";

public:
	void Attack(Hero* p);
	void Hurted();
};

class Yi;    // 为避免互相引用所必要的前置声明

class Garen :public Hero {
	const string name = "Garen";
};

class Yi :public Hero {
	const string name = "Yi";
};
```

```Hero.cpp```：

```c++
#include<iostream>
#include "Hero.h"
using namespace std;

void Hero::Attack(Hero* p) {
	cout << this->name << " attacks!" << endl;
	p->Hurted();
}

void Hero::Hurted(void) {
	cout << this->name << " was attacked!" << endl;
}
```

修改后的程序输出为：

>Hero attacks!\
Hero was attacked!\
Hero attacks!\
Hero was attacked!

继承重载未生效。

### 解决



## 前置声明

这是在写示例时遇到的一个坑。在声明英雄类时，英雄类之间产生了互相引用的问题，结果就是编译器一直报[C2061](https://docs.microsoft.com/zh-cn/cpp/error-messages/compiler-errors-1/compiler-error-c2061?f1url=https%3A%2F%2Fmsdn.microsoft.com%2Fquery%2Fdev16.query%3FappId%3DDev16IDEF1%26l%3DZH-CN%26k%3Dk(C2061)%26rd%3Dtrue%26f%3D255%26MSPPError%3D-2147217396&view=vs-2019)错误。原因就在于如下代码：

```c++
class Garen :public Hero {
	...
public:
	void Attack(Yi* p);
	...
};

class Yi :public Hero {
	...
public:
	void Attack(Garen* p);
	...
};
```

这段代码不管是写在一个文件中还是分开写在多个文件中，都存在互相引用的问题。解决此问题的方式是使用前置引用：

```c++
class Yi;    // 为避免互相引用所必要的前置声明

class Garen :public Hero {
	...
public:
	void Attack(Yi* p);
	...
};

class Yi :public Hero {
	...
public:
	void Attack(Garen* p);
	...
};
```
