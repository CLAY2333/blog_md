---
title: Leetcode刷题之路#48-Rotate Image
date: 2019-04-15 15:08:44
categories: 算法
tags: [矩阵,翻转,数学]

---

<Excerpt in index | 首页摘要> 

翻转二维矩阵，题目限制为不能使用其他的矩阵进行翻转后再赋值，只能在原矩阵上进行翻转。

<!-- more -->

<The rest of contents | 余下全文>

## 思路

首先想到的就是找到原型和旋转90度之后的直接数学规律，但是研究了半天后发现，好像没有直接的转换的数学公式（也有可能是我太菜的没找到2333），然后就想可不可以先进行一次二次转换，于是就寻找了起来，然后发现可以先将原矩阵进行对角线翻转后，再上下翻转，这样就得到了向右90度的翻转了。同理也可得到向左。

## 过程

### 第一次旋转

这次旋转是最具有挑战性的，出错最多的也是这步了，想到后反过来看其实发现也不难

* 将n-1减去当前的y坐标，再减去当前行所在的x坐标(其实这个坐标应该是当前行和对角线交叉的元素的y坐标)。再将需要翻转的左边x,y，都加上这个值，得到坐标位置就是对应的翻转位置了
* 然后还需要注意的就是，其实整个矩阵，只需要翻转的仅仅是对角线左侧的元素

### 第二次翻转

这次翻转就很简单了，直接一个for循环，依次换过来就好了

## 代码

### Rotate Image ⭐️⭐️

- 题目地址/Problem Url: [https://leetcode-cn.com/problems/rotate-image](https://leetcode-cn.com/problems/rotate-image)
- 执行时间/Runtime: 52 ms 
- 内存消耗/Mem Usage: 13.3 MB
- 通过日期/Accept Datetime: 2019-04-15 15:00

```python
// Author: CLAY2333 @ https://github.com/CLAY2333/CLAYleetcode
class Solution:
    def rotate(self, matrix: list) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n=len(matrix[0])
        temp_value=0
        times=0
        for index_x,value_x in enumerate(matrix):
            times=0
            for index_y,value_y in enumerate(value_x):
                if(times>=n-index_x-1):
                    break
                else:
                    x=n-1-index_x-index_y
                    temp_value=matrix[index_x][index_y]
                    matrix[index_x][index_y]=matrix[x+index_x][x+index_y]
                    matrix[x+index_x][x+index_y]=temp_value
                    times+=1
        times=0
        for index_x,value_x in enumerate(matrix):
            if times>=n-index_x:
                break
            temp_value=value_x
            matrix[index_x]=matrix[n-index_x-1]
            matrix[n - index_x-1]=temp_value
            times+=1
```