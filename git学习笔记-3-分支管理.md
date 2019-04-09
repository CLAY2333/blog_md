---
title: git学习笔记(3)-分支管理
date: 2017-02-12 21:26:12
categories: 笔记
tags: [笔记,git,版本控制]
---

<Excerpt in index | 首页摘要> 

>分支就是科幻电影里面的平行宇宙，当你正在电脑前努力学习Git的时候，另一个你正在另一个平行宇宙里努力学习SVN。
>
>如果两个平行宇宙互不干扰，那对现在的你也没啥影响。不过，在某个时间点，两个平行宇宙合并了，结果，你既学会了Git又学会了SVN！
>
>分支在实际中有什么用呢？假设你准备开发一个新功能，但是需要两周才能完成，第一周你写了50%的代码，如果立刻提交，由于代码还没写完，不完整的代码库会导致别人不能干活了。如果等代码全部写完再一次提交，又存在丢失每天进度的巨大风险。
>
>现在有了分支，就不用怕了。你创建了一个属于你自己的分支，别人看不到，还继续在原来的分支上正常工作，而你在自己的分支上干活，想提交就提交，直到开发完毕后，再一次性合并到原来的分支上，这样，既安全，又不影响别人工作。

![pic](http://www.liaoxuefeng.com/files/attachments/001384908633976bb65b57548e64bf9be7253aebebd49af000/0)

<!-- more -->

<The rest of contents | 余下全文>

## 创建与合并分支

### 图文讲解

* 一开始的时候git是这样的

  ![pic](http://www.liaoxuefeng.com/files/attachments/0013849087937492135fbf4bbd24dfcbc18349a8a59d36d000/0)

* 你每次提交的时候master分支都会往前移动一步当我们创建新的分支比如dev的时候,git就会创建一个指针叫dev指向master相同的提交，再把HEAD指向dev，就表示当前分支在dev上

  ![pic](http://www.liaoxuefeng.com/files/attachments/001384908811773187a597e2d844eefb11f5cf5d56135ca000/0)

* 这个时候你再提交的时候也是这个dev的分在动了

  ![pic](http://www.liaoxuefeng.com/files/attachments/0013849088235627813efe7649b4f008900e5365bb72323000/0)

* 当我们完成在dev上的工作的时候就可以吧dev合并到master上了，最简单的就是把master指向dev的当前提交了，完成合并

  ![](http://www.liaoxuefeng.com/files/attachments/00138490883510324231a837e5d4aee844d3e4692ba50f5000/0)

* 合并完之后还可以删除dev指针

  ![](http://www.liaoxuefeng.com/files/attachments/001384908867187c83ca970bf0f46efa19badad99c40235000/0)



### 实战

* 创建dev分支，并且切换到dev分支

  ```Git
  $ git checkout -b dev
  Swiched to a new branch 'ded'
  ```

  加上-b这个参数相当于创建并且切换，等于这两个命令

  ~~~
  $ git branch dev
  $ git checkout dev
  Switched to branch 'dev'
  ~~~

* 然后可以用git branch 去查看所有的分支，他会列出所有的分支，并且在当前分支的前面标注一个*号，然后我们可以在dev分支上正常的提交

* 分支工作完成之后我们可以切回master

  ~~~
  git checkout master
  ~~~

* 切回来之后你会发现你在之前dev上的修改在你的库里面没有发现，因为你库里面是master分支还没合并过来，现在我们把这个合并过来

  ~~~
  git merge dev
  ~~~

  这个命令是用于合并到当前分支，合并之后再查看就可以看到和dev分支的最新提交是一样的

  之后就可以放心的删除dev分支了 git branch -d dev

  删除之后查看就可以看到之后master分支了


### 小结

Git鼓励大量使用分支：

查看分支：`git branch`

创建分支：`git branch `

切换分支：`git checkout `

创建+切换分支：`git checkout -b `

合并某分支到当前分支：`git merge `

删除分支：`git branch -d `



## 解决冲突

我们合并分支的时候也不都是这么一帆风顺

1. 比如你准备了一个新的分支featurel然后你再这个新的分支上开发

   ~~~
   git checkout -b featurel
   ~~~

2. 修改readme.md最后一行

3. 在featurel分支上提交

   ~~~
   git add readme.md
   git commit -m "AND simple"
   ~~~

4. 切换到master分支上

   ~~~
   git checkout master
   ~~~

   git还会自动提示我们，当前分支master分支比远程的master分支要提前一个提交

   在master分支上把readme.txt文件最后一行修改了，提交

   ~~~
   git add master
   git commit -m "&&&"
   ~~~

5. 现在master和featurel都有各自的提交：

   ![](http://www.liaoxuefeng.com/files/attachments/001384909115478645b93e2b5ae4dc78da049a0d1704a41000/0)

6. 这种状态git没办法进行快速合并，只有把各自的修改合并起来这样的合并就会有可能产生冲突

   ~~~
   git merge featurel 
   Auto-merging readme.txt
   CONFLICT (content): Merge conflict in readme.txt
   Automatic merge failed; fix conflicts and then commit the result.
   ~~~

   Git 告诉我们readme.txt文件存在冲突需要手动解决之后才能再提交 使用`git status` 也可以告诉我们冲突的文件内容

7. Git用`<<<<<<<`，`=======`，`>>>>>>>`标记出不同分支的内容

   ~~~
   Git is a distributed version control system.
   Git is free software distributed under the GPL.
   Git has a mutable index called stage.
   Git tracks changes of files.
   <<<<<<< HEAD
   Creating a new branch is quick & simple.
   =======
   Creating a new branch is quick AND simple.
   >>>>>>> feature1
   ~~~

   修改然后保存提交，就变成了这样：

   ![](http://www.liaoxuefeng.com/files/attachments/00138490913052149c4b2cd9702422aa387ac024943921b000/0)

8. 也可以用git log查看分支的合并情况

   ~~~
   git log --graph --pretty=oneline --abbrev-commit
   *   59bc1cb conflict fixed
   |\
   | * 75a857c AND simple
   * | 400b400 & simple
   |/
   * fec145a branch test
   ...
   ~~~

   最后删除featurel分支

   ~~~
   git branch -d featurel
   ~~~


### 小结

当Git无法自动合并分支时，就必须首先解决冲突。解决冲突后，再提交，合并完成。

用`git log --graph`命令可以看到分支合并图。

## 分支管理策略

 通常合并分支的时候，如果可能git会使用Fast forward模式，但是在这种模式下删除分支会丢掉分支的信息，如果强制禁用Fast forward的话，git会在merge时生成一个新的commit，这样在分支历史上就可以看出分支信息了

然后我们实际测试一次 `--no-ff`方式的`git mergr`:

1. 首先我们仍然创建并切换`dev`分支

   ~~~
   $ git checkout -b dev
   ~~~

2. 修改readme.md文件，并提交一个新的commit:

   ~~~
   $ git add readme.md
   $ git commit -m "add merge"
   ~~~

   切回master

   ~~~
   $ git checkout master
   ~~~

3. 准备切回dev分支，注意`--no-ff`参数，表示禁用`Fastforward`:

   ~~~
   $ git merge --no-ff -m "merge with no-ff" dev
   ~~~

4. 因为要合并和创建一个新的commit，所以加上-m 参数，把commit描述写进去，合并后，我们要用 `git log`看看分支历史

   ~~~
   $ git log --graph --pretty=oneline --abbrev-commit
   *   7825a50 merge with no-ff
   |\
   | * 6224937 add merge
   |/
   *   59bc1cb conflict fixed
   ...
   ~~~

可以看到，不使用`Fastforward`模式，，merge后就像这样：

![](http://www.liaoxuefeng.com/files/attachments/001384909222841acf964ec9e6a4629a35a7a30588281bb000/0)

### 分支策略

在实际开发中我们应该按照几个基本的原则进行分支管理

首先master分支应该是十分稳定的，也就是仅用来发布新版本，平时不能在上面干活，一般都是在dev分支上干活，之后觉得版本稳定了在吧dev上的合并到master上。

你和你的小伙伴们每个人都在dev分支上干活，每个人都有自己的分支，时不时地往dev分支上合并就可以了。

所以，团队合作的分支看起来就像这样：

![](http://www.liaoxuefeng.com/files/attachments/001384909239390d355eb07d9d64305b6322aaf4edac1e3000/0)

### 小结：

Git分支十分强大，在团队开发中应该充分应用。

合并分支时，加上`--no-ff`参数就可以用普通模式合并，合并后的历史有分支，能看出来曾经做过合并，而`fast forward`合并就看不出来曾经做过合并。



## Bug分支

软件开发，bug就像家常便饭一样，有了bug就需要修复,在git中，由于分支是如此的强大，所以我们可以每次都开一个分支去修复bug，比如你接受到一个代号101的bug任务的时候，很自然的你想创建一个分支issue-101来修复它，但是等等dev上的工作还没提交。

但是你并不想提交，但是工作只进行到一半的时候，还没办法提交，然而你短时间之内并不能完成。但是必须在两个小时内修复该bug，git提供了stash功能，可以吧当前的工作现场储藏起来，等以后回复现场工作。

~~~
$ git stash
~~~

然后用git status查看工作区，就会发现工作区还是特别干净的，这个时候就可以放心的去创建分支了。

假如你是需要在`master`分支上修复，就从`maste`创建临时分支，然后在分支上吧你的bug修复了吧，然后切回`master`

完成合并，最后删除issue-101，然后是时候返回dev分支干活了

然后你就会发现你的工作区蛮干净的，那么