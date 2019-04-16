---
title: Leetcode刷题之路#49.Group Anagrams
date: 2019-04-16 10:39:15
categories: 算法
tags: [算法,字符串,leetcode,哈希表]
---

<Excerpt in index | 首页摘要> 

leetcode 49题

<!-- more -->

<The rest of contents | 余下全文>

## 思路

思路很简单，就是简单的排序查重就好了，也不准备改进了

## 代码

### Group Anagrams ⭐️⭐️

- 题目地址/Problem Url: <https://leetcode-cn.com/problems/group-anagrams>
- 执行时间/Runtime: 240 ms
- 内存消耗/Mem Usage: 16.1 MB
- 通过日期/Accept Datetime: 2019-04-15 16:44

```python
// Author: CLAY2333 @ https://github.com/CLAY2333/CLAYleetcode
class Solution:
    def groupAnagrams(self, strs:list) -> list:
        D={}
        re=[]

        for index,value in enumerate(strs):
            temp_value=value
            now_value=''.join(sorted(temp_value))
            if D.get(now_value)==None:
                D[now_value]=len(re)
                re.append([value])
            else:
                re[D.get(now_value)].append(value)
        return re
```