---
title: PyTorch学习(1)-优化器
date: 2019-04-23 17:00:27
categories: 深度学习
tags: [PyTorch,深度学习,优化器]
---

<Excerpt in index | 首页摘要> 

关于一些PyTorch框架的学习，不仅仅是关于一些框架的使用，更多的我是想了解一些框架的源码，以便于我可以更加理解其中的缘由。

<!-- more -->

<The rest of contents | 余下全文>

## 优化器

先简单介绍一下优化器(Optimizer),其实优化器就是在变化得时候找个方向和距离，看似很简单，但是对于深度学习非常的重要，因为往哪走走几步对于训练过程是非常重要的，有时候一步走错，导致的错误是连环的。引用[Maddock](http://www.cnblogs.com/adong7639/)的话：

> 机器学习界有一群炼丹师，他们每天的日常是：
>
> 拿来药材（数据），架起八卦炉（模型），点着六味真火（优化算法），就摇着蒲扇等着丹药出炉了。
>
> 不过，当过厨子的都知道，同样的食材，同样的菜谱，但火候不一样了，这出来的口味可是千差万别。火小了夹生，火大了易糊，火不匀则半生半糊。
>
> 机器学习也是一样，模型优化算法的选择直接关系到最终模型的性能。有时候效果不好，未必是特征的问题或者模型设计的问题，很可能就是优化算法的问题。
>
> 说到优化算法，入门级必从 SGD 学起，老司机则会告诉你更好的还有 AdaGrad / AdaDelta，或者直接无脑用 Adam。可是看看学术界的最新 paper，却发现一众大神还在用着入门级的 SGD，最多加个 Momentum 或者 Nesterov，还经常会黑一下Adam。比如 UC Berkeley 的一篇论文就在 Conclusion 中写道：
>
> > *Despite the fact that our experimental evidence demonstrates that adaptive methods are not advantageous for machine learning, the Adam algorithm remains incredibly popular. We are not sure exactly as to why ……*
>
> 无奈与酸楚之情溢于言表。
>
> 这是为什么呢？难道平平淡淡才是真？
>
> 深度学习优化算法经历了 SGD -> SGDM -> NAG ->AdaGrad -> AdaDelta -> Adam -> Nadam 这样的发展历程

我也就主要书写几种我用过的优化器：

## PyTorch中的Optimizer函数

因为我主要是使用的是PyTorch框架，所以在这里介绍一下PyTorch中的优化器的函数，参考余霆嵩的PyTorch实用教程

### 常用方法

* zero_grad() #梯度清零
* step(closure) #进行一次权值更新
* state_dict() #获取模型当前的参数
* load_state_dict(state_dict) #将state_dict 函数的参数加载进入网络
* add_param_group(param_group) #给optimizer管理的参数组中增加一组参数

### 例子：

~~~python
        for step, (data,label) in enumerate(train_loader):
            if step<1000:
                continue
            data = data.to(device)
            label = label.to(device)
            output=net(data)
            loss=criterion(output,label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
~~~

## torch.optim.SGD

~~~
class torch.optim.SGD(params, lr=<object object>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
~~~

随机梯度下降算法，可以说是用得最多也最简单的一种优化器了，虽然简单，但是对于大部分的网络都是用的。

### 参数：

* params(iterable)，优化器要管理的那部分参数。
* lr(float)- 初始学习率，可按需随着训练过程不断调整学习率。momentum(float)- 动量，通常设置为0.9，0.8
* dampening(float)- dampening for momentum ，暂时不了其功能，在源码中是这样用的：buf.mul_(momentum).add_(1 -dampening, d_p)，值得注意的是，若采用nesterov，dampening必须为 0.
* weight_decay(float)- 权值衰减系数，也就是L2正则项的系数
* nesterov(bool)- bool选项，是否使用NAG(Nesterov accelerated gradient)

### 备注

PyTorch的SGD和别的有所不同

Torch：

​			v = \rho * v + g \\
​			p = p - lr * v

其他框架：
			v=ρ∗v+lr∗g
			p=p−v = p - ρ∗v - lr∗g

ρ是动量，v是速率，g是梯度，p是参数，其实差别就是在ρ∗v这一项，Torch中将此项也乘了一个学习率。

## torch.optim.ASGD

ASGD也成为SAG，均表示随机平均梯度下降(Averaged Stochastic Gradient Descent)，简单地说ASGD就是用空间换时间的一种SGD。

* params(iterable)- 参数组，优化器要优化的那些参数。
* lr(float)- 初始学习率，可按需随着训练过程不断调整学习率。
* lambd(float)- 衰减项，默认值1e-4。 
* alpha(float)- power for eta update ，默认值0.75。
* t0(float)- point at which to start averaging，默认值1e6。
* weight_decay(float)- 权值衰减系数，也就是L2正则项的系数。

## torch.optim.Rprop

实现Rprop优化方法(弹性反向传播)，该优化方法适用于full-batch，不适用于mini-batch，因而在mini-batch大行其道的时代里，很少见到。

## torch.optim.Adagrad

实现Adagrad优化方法(Adaptive Gradient)，Adagrad是一种自适应优化方法，是自适应的为各个参数分配不同的学习率。这个学习率的变化，会受到梯度的大小和迭代次数的影响。梯度越大，学习率越小；梯度越小，学习率越大。缺点是训练后期，学习率过小，因为Adagrad累加之前所有的梯度平方作为分母。

## torch.optim.Adadelta

实现Adadelta优化方法。Adadelta是Adagrad的改进。Adadelta分母中采用距离当前时间点比较近的累计项，这可以避免在训练后期，学习率过小。

## torch.optim.RMSprop

实现RMSprop优化方法（Hinton提出），RMS是均方根（root meam square）的意思。RMSprop和Adadelta一样，也是对Adagrad的一种改进。RMSprop采用均方根作为分母，可缓解Adagrad学习率下降较快的问题，并且引入均方根，可以减少摆动。

## torch.optim.Adam(AMSGrad)

实现Adam(Adaptive Moment Estimation))优化方法。Adam是一种自适应学习率的优化方法，Adam利用梯度的一阶矩估计和二阶矩估计动态的调整学习率。吴老师课上说过，Adam是结合了Momentum和RMSprop，并进行了偏差修正。

* amsgrad- 是否采用AMSGrad优化方法，asmgrad优化方法是针对Adam的改进，通过添加额外的约束，使学习率始终为正值

## torch.optim.Adamax

实现Adamax优化方法。Adamax是对Adam增加了一个学习率上限的概念，所以也称之为Adamax。

## torch.optim.SparseAdam

针对稀疏张量的一种“阉割版”Adam优化方法。

## torch.optim.LBFGS

实现L-BFGS（Limited-memory Broyden–Fletcher–Goldfarb–Shanno）优化方法。L-BFGS属于拟牛顿算法。L-BFGS是对BFGS的改进，特点就是节省内存。








