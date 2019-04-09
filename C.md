---
title: 程序中的代码段,BBS段,数据段,堆和栈小解
date: 2016-09-14 20:55:57
categories: 编程
tags: [C++,code,编程]
---

<Excerpt in index | 首页摘要> 

<!-- more -->

<The rest of contents | 余下全文>



## 首先是百度百科的解释:



> 代码段这部分区域的大小在程序运行前就已经确定，并且内存区域通常属于只读, 某些架构也允许代码段为可写，即允许自修改程序。 在代码段中，也有可能包含一些只读的常数变量，例如字符串常量等。后面的解释就是对于VS2005的使用的?Snippet的解释我就没保存的

## 再看一看在进程对应的内存空间中的各个段是干什么的:



BSS段：BSS段（bss segment）通常是指用来存放程序中未初始化的全局变量的一块内存区域。BSS是英文Block Started by Symbol的简称。BSS段属于静态内存分配。
数据段：数据段（data segment）通常是指用来存放程序中已初始化的全局变量的一块内存区域。数据段属于静态内存分配。
代码段：代码段（code segment/text segment）通常是指用来存放程序执行代码的一块内存区域。这部分区域的大小在程序运行前就已经确定，并且内存区域通常属于只读, 某些架构也允许代码段为可写，即允许修改程序。在代码段中，也有可能包含一些只读的常数变量，例如字符串常量等。
堆（heap）：堆是用于存放进程运行中被动态分配的内存段，它的大小并不固定，可动态扩张或缩减。当进程调用malloc等函数分配内存时，新分配的内存就被动态添加到堆上（堆被扩张）；当利用free等函数释放内存时，被释放的内存从堆中被剔除（堆被缩减）
栈(stack)：栈又称堆栈， 是用户存放程序临时创建的局部变量，也就是说我们函数括弧“{}”中定义的变量（但不包括static声明的变量，static意味着在数据段中存放变量）。除此以外，在函数被调用时，其参数也会被压入发起调用的进程栈中，并且待到调用结束后，函数的返回值也会被存放回栈中。由于栈的先进后出特点，所以栈特别方便用来保存/恢复调用现场。从这个意义上讲，我们可以把堆栈看成一个寄存、交换临时数据的内存区。
下图是APUE中的一个典型C内存空间分布图

程序中的代码段,BBS段,数据段,堆和栈小解
例如：

```c
#include <stdio.h>
int g1=0, g2=0, g3=0;
int max(int i){
  int m1=0,m2,m3=0,*p_max;
  static n1_max=0,n2_max,n3_max=0;
  p_max = (int*)malloc(10);
  printf("打印max程序地址\n");
  printf("in max: 0x%08x\n\n",max);
  printf("打印max传入参数地址\n");
  printf("in max: 0x%08x\n\n",&i);
  printf("打印max函数中静态变量地址\n");
  printf("0x%08x\n",&n1_max); //打印各本地变量的内存地址
  printf("0x%08x\n",&n2_max);
  printf("0x%08x\n\n",&n3_max);
  printf("打印max函数中局部变量地址\n");
  printf("0x%08x\n",&m1); //打印各本地变量的内存地址
  printf("0x%08x\n",&m2);
  printf("0x%08x\n\n",&m3);
  printf("打印max函数中malloc分配地址\n");
  printf("0x%08x\n\n",p_max); //打印各本地变量的内存地址
  if(i) return 1;
  else return 0;
}
int main(int argc, char **argv){
  static int s1=0, s2, s3=0;
  int v1=0, v2, v3=0;
  int *p;
  p = (int*)malloc(10);
  printf("打印各全局变量(已初始化)的内存地址\n");
  printf("0x%08x\n",&g1); //打印各全局变量的内存地址
  printf("0x%08x\n",&g2);
  printf("0x%08x\n\n",&g3);
  printf("======================\n");
  printf("打印程序初始程序main地址\n");
  printf("main: 0x%08x\n\n", main);
  printf("打印主参地址\n");
  printf("argv: 0x%08x\n\n",argv);
  printf("打印各静态变量的内存地址\n");
  printf("0x%08x\n",&s1); //打印各静态变量的内存地址
  printf("0x%08x\n",&s2);
  printf("0x%08x\n\n",&s3);
  printf("打印各局部变量的内存地址\n");
  printf("0x%08x\n",&v1); //打印各本地变量的内存地址
  printf("0x%08x\n",&v2);
  printf("0x%08x\n\n",&v3);
  printf("打印malloc分配的堆地址\n");
  printf("malloc: 0x%08x\n\n",p);
  printf("======================\n");
  max(v1);
  printf("======================\n");
  printf("打印子函数起始地址\n");
  printf("max: 0x%08x\n\n",max);
  return 0;
}
```

