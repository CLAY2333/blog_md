---
title: Leetcode刷题之路#46.Permutations
date: 2019-04-12 10:45:44
categories: 算法
tags: [回溯算法,排列组合,递归]
---

<Excerpt in index | 首页摘要> 

首先想到就是用递归算法去完成这个算法，因为没有重复的元素，所以我们不用考虑重复排列的情况，因为我们只需要取出一个加入临时list后再主list去除掉这个value就好了。

<!-- more -->

<The rest of contents | 余下全文>

写了一个子函数用于递归，其他的就很简单了，没啥好讲的

### Permutations ⭐️⭐️

- 题目地址/Problem Url: [https://leetcode-cn.com/problems/permutations](https://leetcode-cn.com/problems/permutations)
- 执行时间/Runtime: 72 ms 
- 内存消耗/Mem Usage: 13.3 MB
- 通过日期/Accept Datetime: 2019-04-12 10:39

```python
// Author: CLAY2333 @ https://github.com/CLAY2333/CLAYleetcode
class Solution:
    def loop(self,nums:list,re:list,now_re:list):
        for index,value in enumerate(nums):
            now_re.append(nums.pop(index))
            if(len(nums)==0):
                re.append(now_re.copy())
                now_re.pop(now_re.index(value))
                nums.insert(index, value)
                return 0
            self.loop(nums,re,now_re)
            now_re.pop(now_re.index(value))
            nums.insert(index,value)
    def permute(self, nums: list) -> list:
        re=[]
        now_re=[]
        self.loop(nums,re,now_re)
        return re

```

