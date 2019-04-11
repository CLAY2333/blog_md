---
title: #43Leetcode刷题之路-Multiply Strings
date: 2019-04-11 15:44:54
categories: 算法
tags: [算法,数据结构,字符串,乘法,数学]
---

<Excerpt in index | 首页摘要> 

目的是为了实现字符串的乘法，自然也不可能是简单的使用toint之后再相乘，并且还应该考虑到大数的情况。

<!-- more -->

<The rest of contents | 余下全文>

# 思路

首先我们可以想到的就是直接模拟乘法的功能，将字符串的每一位相乘，然后进位。代码如下：

~~~python
Multiply Strings ⭐️⭐️
题目地址/Problem Url: https://leetcode-cn.com/problems/multiply-strings
执行时间/Runtime: 772 ms
内存消耗/Mem Usage: 13.3 MB
通过日期/Accept Datetime: 2019-04-11 11:44
class Solution:
    def Add(self,S1,S2,num):
        if(len(S2)==0):
            return S1
        for zero in range(num-1):
            S1.append('0')
        on=0
        re=[]
        S2.reverse()
        S1.reverse()
        for index,value in enumerate(S1):
            if(index<len(S2)):
                temp=int(value)+int(S2[index])+on
                on=temp//10
                re.append(str(temp%10))
            else:
                temp=int(value)+on
                re.append(str(temp%10))
                on=temp//10
        if(on!=0):
            re.append(str(on))
        re.reverse()
        return re

    def multiply(self, num1: str, num2: str) -> str:
        if(num1=='0' or num2=='0'):
            return '0'
        re=[]
        temp_re=[]
        on=0
        num=1
        num2=list(num2)
        num1=list(num1)
        num1.reverse()
        num2.reverse()
        for index_2,value_2 in enumerate(num2):
            on = 0
            temp_re=[]
            for index_1,value_1 in enumerate(num1):
                temp=int(value_2)*int(value_1)+on
                on=temp//10
                temp_re.insert(0,str(temp%10))
            if(on!=0):
                temp_re.insert(0,str(on))
            re=self.Add(temp_re,re,num)
            num+=1
        return ''.join(re)
~~~

提交后遇到过一些错误，首先就是因为我是将进位保存到on这个字段，那么如果在乘法的最后一位还存在进位的话，这个进位就会被丢掉，导致错误，解决的办法有两种：

1. 就是在循环结束后，检查on的状态，如果非0，就需要额外添加进位
2. 放弃使用on字段记录进位，因为在乘法来看，进位仅可能在前一位上，不可能在其他地方，所以可以直接先将进位加到结果上即可。

此方法速度有待改进。有考虑在之后进行改进