---
title: C++的回调函数
date: 2016-09-16 22:02:44
categories: 编程
tags: [c++,code,编程]
---

<Excerpt in index | 首页摘要> 

回调函数，顾名思义，就是使用者自己定义一个函数，使用者自己实现这个函数的程序内容，然后把这个函数作为参数传入别人（或系统）的函数中，由别人（或系统）的函数在运行时来调用的函数。函数是你实现的，但由别人（或系统）的函数在运行时通过参数传递的方式调用，这就是所谓的回调函数。简单来说，就是由别人的函数运行期间来回调你实现的函数。

<!-- more -->

<The rest of contents | 余下全文>

## 函数指针

### 概览

指针是一个变量,是用来指向内存地址的.一个程序运行的时候是要把所有东西放到内存中的(这个在我的之前的blog里面讲过了,有好几个段去存),这就决定了程序运行的时候是可以去指向他们的.函数是存放在内存代码区域内的,他们也有内存地址所以可以用指针去指函数这种指向函数入口地址的指针成为函数指针

## 函数调用的代码

```c++
#include<iostream>
#include <string>
using namespace std;
void callback(char *s){
    cout<<s<<endl;
 }
int main(){
    callback("sss");
    return 0;
}
```

## 函数指针的代码

```c++
#include<iostream>
#include <string>
using namespace std;

void callback(char *s){
    cout<<s<<endl;
 }
int main(){
    void (*f)(char *s);
    f=callback;
    f("23333");
    return 0;
}

```

上面的例子可以看出来,调用是直接把子函数放在int main 里面直接运行,这样的使用方式就是调用.但是用函数指针的话就是初始一个指针去指向这个函数的入口,然后通过这个函数去传递参数运行这个函数.

### 用宏定义

```c++
#include<iostream>
#include <string>
using namespace std;

typedef void (*F)(char* s);
void callback(char *s){
    cout<<s<<endl;
 }
int main(){
    F f;
    f=callback;
    f("23333");
    return 0;
}

```

### 函数指针数组

```c++
#include<iostream>
#include <string>
using namespace std;

typedef void (*F)(char* s);
void f1(char* s){cout<<s<<1;}
void f2(char* s){cout<<s<<2;}
void f3(char* s){cout<<s<<3;}
void callback(char *s){
    cout<<s<<endl;
 }
int main(){
    F f[]={f1,f2,f3};
    f[1]("lzy");
    return 0;
}

```



## 回调函数

### 概念

回调函数，顾名思义，就是使用者自己定义一个函数，使用者自己实现这个函数的程序内容，然后把这个函数作为参数传入别人（或系统）的函数中，由别人（或系统）的函数在运行时来调用的函数。函数是你实现的，但由别人（或系统）的函数在运行时通过参数传递的方式调用，这就是所谓的回调函数。简单来说，就是由别人的函数运行期间来回调你实现的函数.

```c++
#include<iostream>
#include <string>
using namespace std;
void Print(char *s){
    cout<<s<<endl;
}
void CallPrint(void (*f)(char*),char* s){
    f(s);
}
int main(){
    CallPrint(Print,"2333");
    return 0;
}

```

表面上我们是将这个指针改成了函数,但是我们只可以只提供一个函数的入口,将这函数拿给别人去运行,这样的话就可将这个入口给别人来运行你写的代码了