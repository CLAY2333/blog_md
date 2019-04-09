---
title: Callback学习记录(未完)
date: 2016-09-21 22:17:25
categories: 编程
tags: [c++,code,编程]
---

<Excerpt in index | 首页摘要> 

前几天dalao开了一个公开课,我去听了听回调函数,对于这个原理我是基本理解了,大概就是我之前的那个blog那种方式,但是其中的很多的晦涩的函数和语法我决定记下来,这篇文章就是记录的,代码我会在后面贴出来.

<!-- more -->

<The rest of contents | 余下全文>

## 笔记

### class和typename的区别

在C++中很多地方都用到了这两,class用于定义类，在模板引入c++后，最初定义模板的方法为： template<class T> 在这里class关键字表明T是一个类型，后来为了避免class在这两个地方的使用可能给人带来混淆，所以引入了typename这个关键字，它的作用同class一样表明后面的符号为一个类型，这样在定义模板的时候就可以使用下面的方式了： template<typenameT>,在模板定义语法中关键字class与typename的作用完全一样。typename难道仅仅在模板定义中起作用吗？其实不是这样，typename另外一个作用为：使用嵌套依赖类型(nested depended name)，如下所示：

```c++
class MyArray 
{ 
public：
typedef int LengthType;
.....
}

template<class T>
void MyMethod( T myarr ) 
{ 
typedef typename T::LengthType LengthType; 
LengthType length = myarr.GetLength; 
}
```

这个时候typename的作用就是告诉c++编译器，typename后面的字符串为一个类型名称，而不是成员函数或者成员变量，这个时候如果前面没有typename，编译器没有任何办法知道T::LengthType是一个类型还是一个成员名称(静态数据成员或者静态函数)，所以编译不能够通过。

### 可变参模板(动态模板/长参数模板)

可变参数模板(variadic template)就是一个接受可变参数的模板函数或者模板类,可变的称为参数包,存在两种参数包模板参数包,函数参数包(表示零个或者多个函数参数)

```c++
//Args是一个模板参数包,rest是一个函数参数包
//Args表示零个或者多个模板类些参数
//rest表示零个或多个函数参数
template <typename T, typename... Args>
void foo(const T &t,const Args& ... rest);
```

这样就初始化快了一个可变参模板了,然后看个使用的例子

```c++
int i=0;
double d=3.14;
string s="holle word";
foo(i,s,42,d);
foo(s,42,"hi");
foo(d,s);
foo("hi");
```

完成这种函数的方式是可以用函数重载去完成后的,但是在C11之后我们就可以用可变参函数模板去实现这个功能了.这样这些函数的声明其实就是下面的样子了;

```c++
void foo(const int& , const string&,const int&,const double);
void foo(const string&, const int&,const char(3)&);
void foo(const double&,const string&);
void foo(const char[3]&);
```

在这些中,T就是根据第一个来定的,	剩下的就是根据额外的实参了