---
title: git学习笔记(1)-文件管理
date: 2017-02-02 14:00:27
categories: 笔记
tags: [笔记,git,版本控制]
---

<Excerpt in index | 首页摘要> 

之前做项目是时候一直是使用的是svn但是需要自己搭建服务器，在目前资源紧张的时候使用git是一个不错的选择，自己也在使用github所以决定好好学习一发

<!-- more -->

<The rest of contents | 余下全文>

## 首记

这篇文章是为了记录我学习git的过程而是为了之后方便我翻阅这个学习的过程

## 环境

mac

## 创建版本库

- 首先先创建一个新的文件夹
- 然后在这个路径下通过git init 让这个路径变成版本库
- 之后你需要添加什么文件就git add，然后在用git commit告诉gt你要添加到这个文件到版本库

代码示例：

```
git init
git add readme.md
git commit -m "add readme.md"
```

## 关于版本的控制

你可以查看各个版本的区别

### 修改了本地

- 修改了本地但是还没提交查看那些被修改 git status
- 查看修改内容 git diff

### 版本回退

- 查看最近的提交日志 git log （可选添加后缀 —pretty=oneline）
- 如果你要回退就使用 git reset —hard HEAD^ 这个hard后面跟的就是git的版本号

### 工作区



1. 我们使用git add的时候就是把这个文件添加进入这个暂存区内

   ![0-1](http://www.liaoxuefeng.com/files/attachments/001384907702917346729e9afbf4127b6dfbae9207af016000/0)

2. git commit之后就变成这样了![0-1](http://www.liaoxuefeng.com/files/attachments/0013849077337835a877df2d26742b88dd7f56a6ace3ecf000/0)

总结：简单的就是说add是你的购物车，commit是收银台

### 撤销修改

如果你修改错了某些地方,这个时候你需要撤销你之前进行的修改的话就可以这样：

1. 首先你可以先git status查看你修改了那些文件那些数据

   ```
   $ git status
   # On branch master
   # Changes not staged for commit:
   #   (use "git add <file>..." to update what will be committed)
   #   (use "git checkout -- <file>..." to discard changes in working directory)
   #
   #       modified:   readme.txt
   #
   no changes added to commit (use "git add" and/or "git commit -a")
   ```

   ​

   ​

2. 这个时候你可以用git checkout -- file撤销你的工作区内的修改，比如:

   ```
   git checkout -- readme.txt
   ```

3. 这里有两种状态，一种是`readme.txt`自修改后还没有被放到暂存区，现在，撤销修改就回到和版本库一模一样的状态；一种是`readme.txt`已经添加到暂存区后，又作了修改，现在，撤销修改就回到添加到暂存区后的状态。

   总之，就是让这个文件回到最近一次`git commit`或`git add`时的状态。

   ​

4. 如果是已经被提交到了暂存区这个时候你也可以使用status这个命令


1. git reset 不仅仅可以回退版本也可以吧暂存区的修改回退到工作区

   ```
   git reset HEAD readme.md
   ```

总结：

场景1：当你改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令`git checkout -- file`。

场景2：当你不但改乱了工作区某个文件的内容，还添加到了暂存区时，想丢弃修改，分两步，第一步用命令`git reset HEAD file`，就回到了场景1，第二步按场景1操作。

场景3：已经提交了不合适的修改到版本库时，想要撤销本次提交，参考[版本回退](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/0013744142037508cf42e51debf49668810645e02887691000)一节，不过前提是没有推送到远程库。

### 删除文件

* 我们先添加一个文件

  ```
  git add lll.md
  CLAYhi:repository clay$ git commit -m "add lll.md"
  [master 8c09400] add lll.md
   Committer: CLAY <clay@CLAYhi.local>
  Your name and email address were configured automatically based
  on your username and hostname. Please check that they are accurate.
  You can suppress this message by setting them explicitly. Run the
  following command and follow the instructions in your editor to edit
  your configuration file:

      git config --global --edit

  After doing this, you may fix the identity used for this commit with:

      git commit --amend --reset-author

   2 files changed, 151 insertions(+), 2 deletions(-)
   create mode 100644 lll.md
  CLAYhi:repository clay$ 
  ```

* 一般我们都会直接把文件管理器中的文件给删除了，使用rm lll.md 这个命令，使用之后你可以用git status看看，发现git已经发现你删除了这个文件这个时候你有两个选择。

* 一个是你确定是删除这个文件，你就可以用git rm删除

  ```
  CLAYhi:repository clay$ git rm lll.md
  rm 'lll.md'
  CLAYhi:repository clay$ git commit -m "remove lll.md"
  [master c4c5d04] remove lll.md
   Committer: CLAY <clay@CLAYhi.local>
  Your name and email address were configured automatically based
  on your username and hostname. Please check that they are accurate.
  You can suppress this message by setting them explicitly. Run the
  following command and follow the instructions in your editor to edit
  your configuration file:

      git config --global --edit

  After doing this, you may fix the identity used for this commit with:

      git commit --amend --reset-author

   1 file changed, 149 deletions(-)
   delete mode 100644 lll.md
  ```

  这样在版本库就删除了记得git rm之后还需要commit

* 另一种是你删错了，你需要回复，你也可以恢复，

  ```
  git checkout -- test.txt
  ```

命令`git rm`用于删除一个文件。如果一个文件已经被提交到版本库，那么你永远不用担心误删，但是要小心，你只能恢复文件到最新版本，你会丢失**最近一次提交后你修改的内容**

