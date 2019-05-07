---
title: Leetcode刷题之路#64.Minimum Path Sum
date: 2019-05-06 15:44:54
categories: 算法
tags: [算法,动态规划,矩阵,图]

---

<Excerpt in index | 首页摘要> 

目的是在n*m的矩阵中找到一条从左上角到右下角的路，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

<!-- more -->

<The rest of contents | 余下全文>

# 思路

这道题是非常典型的一个一个动态规划问题，其实问题的解决一个式子就可以解决：

~~~ python
dis[j][i]=min(dis[j-1][i],dis[j][i-1])+grid[j][i]
~~~

因为路径只允许向下和向右走，那么只需要看该点的上一个和左边那个亦可，在选其中的较小值为路径就ok了。最后得到的右下角的肯定就是最小的值。

```python
Minimum Path Sum ⭐️⭐️
题目地址/Problem Url: https://leetcode-cn.com/problems/minimum-path-sum
执行时间/Runtime: 76 ms
内存消耗/Mem Usage: 14.5 MB
通过日期/Accept Datetime: 2019-05-06 20:05
// Author: CLAY2333 @ https://github.com/CLAY2333/CLAYleetcode
class Solution:
    def minPathSum(self, grid: list) -> int:
        dis=[[0 for i in range(len(grid[0]))] for i in range(len(grid))]
        dis[0][0]=grid[0][0]
        for i in range(1,len(grid[0])):
            dis[0][i]=dis[0][i-1]+grid[0][i]
        for i in range(1,len(grid)):
            dis[i][0]=dis[i-1][0]+grid[i][0]
        for i in range(1, len(grid[0])):
            for j in range(1, len(grid)):
                dis[j][i]=min(dis[j-1][i],dis[j][i-1])+grid[j][i]
        return dis[-1][-1]
```

