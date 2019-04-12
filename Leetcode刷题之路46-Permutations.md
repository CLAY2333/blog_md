---
title: Leetcode刷题之路46.Permutations
date: 2019-04-12 10:45:44
categories:
tags:
---

<Excerpt in index | 首页摘要> 



<!-- more -->

<The rest of contents | 余下全文>



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

