---
title: git学习笔记(2)-远程仓库
date: 2017-02-04 16:36:34
categories: 笔记
tags: [笔记,git,版本控制]
---

<Excerpt in index | 首页摘要> 

这篇是为了远程连接上我们的git服务器，因为资源有限我就使用github做示范了

<!-- more -->

<The rest of contents | 余下全文>

## 通过ssh连接github

> 第1步：创建SSH Key。在用户主目录下，看看有没有.ssh目录，如果有，再看看这个目录下有没有`id_rsa`和`id_rsa.pub`这两个文件，如果已经有了，可直接跳到下一步。如果没有，打开Shell（Windows下打开Git Bash），创建SSH Key：
>
> ```
> $ ssh-keygen -t rsa -C "youremail@example.com"
>
> ```
>
> 你需要把邮件地址换成你自己的邮件地址，然后一路回车，使用默认值即可，由于这个Key也不是用于军事目的，所以也无需设置密码。
>
> 如果一切顺利的话，可以在用户主目录里找到`.ssh`目录，里面有`id_rsa`和`id_rsa.pub`两个文件，这两个就是SSH Key的秘钥对，`id_rsa`是私钥，不能泄露出去，`id_rsa.pub`是公钥，可以放心地告诉任何人。
>
> 第2步：登陆GitHub，打开“Account settings”，“SSH Keys”页面：
>
> 然后，点“Add SSH Key”，填上任意Title，在Key文本框里粘贴`id_rsa.pub`文件的内容：
>
> ![pic](http://www.liaoxuefeng.com/files/attachments/001384908342205cc1234dfe1b541ff88b90b44b30360da000/0)
>
> 点“Add Key”，你就应该看到已经添加的Key：
>
> 
>
> ![pic](http://www.liaoxuefeng.com/files/attachments/0013849083502905a4caa2dc6984acd8e39aa5ae5ad6c83000/0)
>
> 为什么GitHub需要SSH Key呢？因为GitHub需要识别出你推送的提交确实是你推送的，而不是别人冒充的，而Git支持SSH协议，所以，GitHub只要知道了你的公钥，就可以确认只有你自己才能推送。
>
> 当然，GitHub允许你添加多个Key。假定你有若干电脑，你一会儿在公司提交，一会儿在家里提交，只要把每台电脑的Key都添加到GitHub，就可以在每台电脑上往GitHub推送了。
>
> 最后友情提示，在GitHub上免费托管的Git仓库，任何人都可以看到喔（但只有你自己才能改）。所以，不要把敏感信息放进去。
>
> 如果你不想让别人看到Git库，有两个办法，一个是交点保护费，让GitHub把公开的仓库变成私有的，这样别人就看不见了（不可读更不可写）。另一个办法是自己动手，搭一个Git服务器，因为是你自己的Git服务器，所以别人也是看不见的。这个方法我们后面会讲到的，相当简单，公司内部开发必备。
>
> 确保你拥有一个GitHub账号后，我们就即将开始远程仓库的学习。
>
> ​								-以上来自廖雪峰的博客

## 添加远程git库

* 现在的情景是，你已经在本地创建了一个Git仓库后，又想在GitHub创建一个Git仓库，并且让这两个仓库进行远程同步，这样，GitHub上的仓库既可以作为备份，又可以让其他人通过该仓库来协作，真是一举多得。

  首先，登陆GitHub，然后，在右上角找到“Create a new repository”按钮，创建一个新的仓库：

  ![pic](http://www.liaoxuefeng.com/files/attachments/0013849084639042e9b7d8d927140dba47c13e76fe5f0d6000/0)

* 在2的位置填写上你的git的库的名字，其他的默认就好了，然后点击Create repository,就完成了

* 现在创建出来了但是这个仓库还是空的我们需要把本地的仓库关联上去

  ```
  git remote add origin git@github.com:@@@/learngit.git
  ```

  需要把@@@改成的github账户名，然后推送本地git push -u origin master接下来的按照提示就完成了，现在你刷新就发现仓库和本地同步了，



## 从远程库克隆

* 上次我们讲了先有本地库，后有远程库的时候，如何关联远程库。

  现在，假设我们从零开发，那么最好的方式是先创建远程库，然后，从远程库克隆。

  首先，登陆GitHub，创建一个新的仓库，名字叫`gitskills`：

  ![pic](http://www.liaoxuefeng.com/files/attachments/0013849085474010fec165e9c7449eea4417512c2b64bc9000/0)

  勾选这个红框部分，这样就可以看到自动生成了一个README.md的文件:

  ![pic](http://www.liaoxuefeng.com/files/attachments/0013849085607106c2391754c544772830983d189bad807000/0)

* 现在仓库有了，下一步是用git clone

  ```
  git clone @@@
  ```

  这个@@@是需要换成项目地址的就ok了。

## 小结

> 要克隆一个仓库，首先必须知道仓库的地址，然后使用`git clone`命令克隆。
>
> Git支持多种协议，包括`https`，但通过`ssh`支持的原生`git`协议速度最快。