---
title: python爬虫日记-下载网页
date: 2016-12-02 23:12:47
categories: 编程
tags: [python,code,爬虫]
---

<Excerpt in index | 首页摘要> 

python在大一的时候就尝试着入门了,前不久突然觉得爬虫是一个挺有趣的东西,所以决定开始学习一下,我也是个新手,所以瞎写下这个也是为了之后查阅更加的方便

<!-- more -->

<The rest of contents | 余下全文>



# 前篇-爬虫的简介



网络爬虫（又被称为网页蜘蛛，网络机器人，在FOAF社区中间，更经常的称为网页追逐者），是一种按照一定的规则，自动地抓取万维网信息的程序或者脚本。另外一些不常使用的名字还有蚂蚁、自动索引、模拟程序或者蠕虫.

# 正文

## 下载网页

首先我们要想着下载访问下载网页,我们使用python的urllib2模块下载

~~~python
import urllib2

def download(url)
return urllib2.urlopen(url).read()
~~~

传入url参数,这个函数会下载网页并且返回html,不过这个简单的代码有个小问题,就是我们可能会遇到这个页面不存在的时候,这个时候这个代码就没办法处理会抛出异常,我们需要做一个更加健壮的版本

~~~python
import urllib2

def download(url):
	print 'Download:',url
    try:
        html =urllib2.urlopen(url).read()
    except urllib2.URLError as e:
        print 'Download error:',e.reason
        html=None
    return html
~~~



捕获到异常的时候就会返回None

但是如果返回的是比如503这种错误我们就可以再次尝试,所以我们需要加这个重新下载功能

```python
import urllib2

def download(url,num_retries=2):
	print 'Download:',url
    try:
        html =urllib2.urlopen(url).read()
    except urllib2.URLError as e:
        print 'Download error:',e.reason
        if num_retries > 0 :
            if hasattr(e,'code') and 500<=e.reason<600:
              return download(url,num-retries-1)
    return html
```

现在当download遇到错误码的时候会判断进行重试了



## 设置用户代理

默认情况下,urllib2使用的是是python2.7作为用户代理下载,但是用可辨识的用户可以让我们避免一些问题,一些网页为了避免爬虫造成的负载,会封禁这个默认用户代理.

```python
import urllib2

def download(url,user_agent='wswp',num_retries=2):
    print 'Downloading:',url
    headers={'User-agent':user_agent}
    request=urllib2.Request(url,headers=headers)
    try:
        html =urllib2.urlopen(url).read()
    except urllib2.URLError as e:
        print 'Download error:', e.reason
        html =None
        if num_retries > 0:
            if hasattr(e, 'code') and 500<=e.code<600:
                return download(user_agent,num_retries-1)
    return html
```

wswp即是Web Scraping with python,