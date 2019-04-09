---
title: How transferable are features in deep neural networks?论文翻译
date: 2018-11-06 14:08:48
categories: 论文翻译
tags: [论文翻译,机器学习,深度学习,迁移学习]
---

<Excerpt in index | 首页摘要> 

该论文比较的特别，因为其实论文中没有任何的代码和算法，整个都是在验证一种实验，但是结果非常有趣的，带给人很大的启发。

<!-- more -->

<The rest of contents | 余下全文>

# 深度神经网络中的特征是否可转移？

## 摘要

Many deep neural networks trained on natural images exhibit a curious phe- nomenon in common: on the first layer they learn features similar to Gabor filters and color blobs. Such first-layer features appear not to be specific to a particular dataset or task, but general in that they are applicable to many datasets and tasks. Features must eventually transition from general to specific by the last layer of the network, but this transition has not been studied extensively. In this paper we experimentally quantify the generality versus specificity of neurons in each layer of a deep convolutional neural network and report a few surprising results. Trans- ferability is negatively affected by two distinct issues: (1) the specialization of higher layer neurons to their original task at the expense of performance on the target task, which was expected, and (2) optimization difficulties related to split- ting networks between co-adapted neurons, which was not expected. In an exam- ple network trained on ImageNet, we demonstrate that either of these two issues may dominate, depending on whether features are transferred from the bottom, middle, or top of the network. We also document that the transferability of fea- tures decreases as the distance between the base task and target task increases, but that transferring features even from distant tasks can be better than using random features. A final surprising result is that initializing a network with transferred features from almost any number of layers can produce a boost to generalization that lingers even after fine-tuning to the target dataset.

在自然图像上训练的许多深度神经网络都表现出一种奇怪的现象：在第一层，他们学习类似于Gabor滤波器和颜色斑点的特征。这样的第一层特征似乎不是特定于特定数据集或任务，而是一般的，因为它们适用于许多数据集和任务。功能必须最终由网络的最后一层从一般转换为特定，但这种转变尚未得到广泛研究。在本文中，我们通过实验量化深度卷积神经网络每层神经元的一般性与特异性，并报告了一些令人惊讶的结果。可转移性受到两个不同问题的负面影响：（1）高层神经元专用于其原始任务，但牺牲了目标任务的性能，这是预期的，以及（2）与分裂网络相关的优化困难在共同适应的神经元之间，这是不期望的。在ImageNet培训的示例网络中，我们证明这两个问题中的任何一个都可能占主导地位，具体取决于功能是从网络的底部，中间还是顶部传输。我们还记录了特征的可转移性随着基本任务和目标任务之间的距离的增加而减少，但即使从远程任务转移特征也可能比使用随机特征更好。最后一个令人惊讶的结果是，从几乎任意数量的层初始化具有传输特征的网络可以促进即使在微调到目标数据集之后仍然存在的泛化。

## 简介

Modern deep neural networks exhibit a curious phenomenon: when trained on images, they all tend to learn first-layer features that resemble either Gabor filters or color blobs. The appearance of these filters is so common that obtaining anything else on a natural image dataset causes suspicion of poorly chosen hyperparameters or a software bug. This phenomenon occurs not only for different datasets, but even with very different training objectives, including supervised image classification (Krizhevsky et al., 2012), unsupervised density learning (Lee et al., 2009), and unsupervised learn- ing of sparse representations (Le et al., 2011).

现代深度神经网络呈现出一种奇怪的现象：当对图像进行训练时，它们都倾向于学习类似于Gabor滤波器或颜色斑点的第一层特征。这些过滤器的外观非常普遍，以至于在自然图像数据集上获取任何其他内容会导致怀疑选择不当的超参数或软件错误。这种现象不仅发生在不同的数据集中，甚至还有非常不同的训练目标，包括监督图像分类（Krizhevsky等，2012），无监督密度学习（Lee et al。，2009），以及无监督学习稀疏表示（Le et al。，2011）。

Because finding these standard features on the first layer seems to occur regardless of the exact cost function and natural image dataset, we call these first-layer features general. On the other hand, we know that the features computed by the last layer of a trained network must depend greatly on the chosen dataset and task. For example, in a network with an N-dimensional softmax output layer that has been successfully trained toward a supervised classification objective, each output unit will be specific to a particular class. We thus call the last-layer features specific. These are intuitive notions of general and specific for which we will provide more rigorous definitions below. If first-layer features are general and last-layer features are specific, then there must be a transition from general to specific somewhere in the network. This observation raises a few questions:

因为无论确切的成本函数和自然图像数据集如何，在第一层上发现这些标准特征似乎都会发生，我们称这些第一层特征是通用的。另一方面，我们知道由训练网络的最后一层计算的特征必须在很大程度上取决于所选择的数据集和任务。例如，在具有已成功训练到监督分类目标的N维softmax输出层的网络中，每个输出单元将特定于特定类。因此，我们将最后一层特征称为特定的。这些是一般和具体的直观概念，我们将在下面提供更严格的定义。如果第一层功能是通用的，而最后一层功能是特定的，则必须在网络中的某个位置从一般到特定的转换。这一观察提出了一些问题：

* Can we quantify the degree to which a particular layer is general or specific?

* Does the transition occur suddenly at a single layer, or is it spread out over several layers? 

* Where does this transition take place: near the first, middle, or last layer of the network?

* 我们能否量化特定层的一般或特定程度？

* 转换是在单个层突然发生，还是在几个层上展开？ 

* 这种转变发生在哪里：靠近网络的第一层，中层或最后一层？

We are interested in the answers to these questions because, to the extent that features within a network are general, we will be able to use them for transfer learning (Caruana, 1995; Bengio et al., 2011; Bengio, 2011). In transfer learning, we first train a base network on a base dataset and task, and then we repurpose the learned features, or transfer them, to a second target network to be trained on a target dataset and task. This process will tend to work if the features are general, meaning suitable to both base and target tasks, instead of specific to the base task. 

我们对这些问题的答案感兴趣，因为，如果网络中的特征是一般的，我们将能够将它们用于转移学习（Caruana，1995; Bengio等，2011; Bengio，2011）。在转移学习中，我们首先在基础数据集和任务上训练基础网络，然后我们将学习的特征重新调整或转移到第二个目标网络，以便在目标数据集和任务上进行训练。如果功能是通用的，这意味着适用于基本任务和目标任务，而不是特定于基本任务，则此过程将起作用。

When the target dataset is significantly smaller than the base dataset, transfer learning can be a powerful tool to enable training a large target network without overfitting; Recent studies have taken advantage of this fact to obtain state-of-the-art results when transferring from higher layers (Donahue et al., 2013a; Zeiler and Fergus, 2013; Sermanet et al., 2014), collectively suggesting that these layers of neural networks do indeed compute features that are fairly general. These results further emphasize the importance of studying the exact nature and extent of this generality. 

当目标数据集明显小于基础数据集时，转移学习可以成为一种强大的工具，可以在不过度拟合的情况下训练大型目标网络;最近的研究利用这一事实来获得从较高层转移时的最新结果（Donahue等，2013a; Zeiler和Fergus，2013; Sermanet等，2014），共同表明这些层神经网络确实计算出相当普遍的特征。这些结果进一步强调了研究这种普遍性的确切性质和程度的重要性。

The usual transfer learning approach is to train a base network and then copy its first n layers to the first n layers of a target network. The remaining layers of the target network are then randomly initialized and trained toward the target task. One can choose to backpropagate the errors from the new task into the base (copied) features to fine-tune them to the new task, or the transferred feature layers can be left frozen, meaning that they do not change during training on the new task. The choice of whether or not to fine-tune the first n layers of the target network depends on the size of the target dataset and the number of parameters in the first n layers. If the target dataset is small and the number of parameters is large, fine-tuning may result in overfitting, so the features are often left frozen. On the other hand, if the target dataset is large or the number of parameters is small, so that overfitting is not a problem, then the base features can be fine-tuned to the new task to improve performance. Of course, if the target dataset is very large, there would be little need to transfer because the lower level filters could just be learned from scratch on the target dataset. We compare results from each of these two techniques — fine-tuned features or frozen features — in the following sections. 

通常的迁移学习方法是训练基础网络，然后将其前n层复制到目标网络的前n层。然后，目标网络的剩余层被随机初始化并针对目标任务进行训练。可以选择将新任务中的错误反向传播到基本（复制）功能中，以将它们微调到新任务，或者可以将传输的要素图层冻结，这意味着它们在新任务的培训期间不会更改。是否微调目标网络的前n层的选择取决于目标数据集的大小和前n层中的参数数量。如果目标数据集很小且参数数量很大，则微调可能会导致过度拟合，因此这些特征通常会被冻结。另一方面，如果目标数据集很大或参数数量很少，那么过度拟合不是问题，那么可以将基本特征微调到新任务以提高性能。当然，如果目标数据集非常大，则几乎不需要迁移，因为可以在目标数据集上从头开始学习较低级别的过滤器。我们将在以下部分中比较这两种技术的结果 - 微调功能或冻结功能。

In this paper we make several contributions:
1. We define a way to quantify the degree to which a particular layer is general or specific,namely, how well features at that layer transfer from one task to another (Section 2). We then train pairs of convolutional neural networks on the ImageNet dataset and characterize the layer-by-layer transition from general to specific (Section 4), which yields the following four results.
2. We experimentally show two separate issues that cause performance degradation when us- ing transferred features without fine-tuning: (i) the specificity of the features themselves, and (ii) optimization difficulties due to splitting the base network between co-adapted neurons on neighboring layers. We show how each of these two effects can dominate at different layers of the network. (Section 4.1)
3. We quantify how the performance benefits of transferring features decreases the more dissimilar the base task and target task are. (Section 4.2)
4. On the relatively large ImageNet dataset, we find lower performance than has been previously reported for smaller datasets (Jarrett et al., 2009) when using features computed from random lower-layer weights vs. trained weights. We compare random weights to transferred weights— both frozen and fine-tuned—and find the transferred weights perform better. (Section 4.3)
5. Finally, we find that initializing a network with transferred features from almost any number of layers can produce a boost to generalization performance after fine-tuning to a new dataset. This is particularly surprising because the effect of having seen the first dataset persists even after extensive fine-tuning. (Section 4.1)

在本文中，我们做了几点贡献：
1. 我们定义了一种方法来量化特定层的一般或特定程度，即该层的特征从一个任务转移到另一个任务的程度（第2节）。然后，我们在ImageNet数据集上训练成对的卷积神经网络，并描述从一般到特定的逐层过渡（第4节），其产生以下四个结果。
2. 我们通过实验显示了两个单独的问题，这些问题在使用传输特征时没有进行微调导致性能下降：（i）特征本身的特殊性，以及（ii）由于在共适应神经元之间分裂基础网络而导致的优化困难在相邻的图层上。我们展示了这两种效应中的每一种如何在网络的不同层面占据主导地位。 （第4.1节）
3. 我们量化了传递特征的性能优势如何降低，基本任务和目标任务的差异越大。 （4.2节）
4. 在相对较大的ImageNet数据集中，我们发现当使用从随机低层权重和训练权重计算的特征时，性能低于之前针对较小数据集报告的性能（Jarrett等，2009）。我们将随机权重与转移的权重进行比较 - 冷冻和微调 - 并发现转移的权重表现更好。 （第4.3节）
5. 最后，我们发现在几乎任意数量的层中初始化具有传输特征的网络可以在微调到新数据集之后提高泛化性能。这尤其令人惊讶，因为即使经过大量微调，看到第一个数据集的效果仍然存在。 （第4.1节）

## Generality vs. Specificity Measured as Transfer Performance 通用性与特异性测量为转移性能

We have noted the curious tendency of Gabor filters and color blobs to show up in the first layer of neural networks trained on natural images. In this study, we define the degree of generality of a set of features learned on task A as the extent to which the features can be used for another task B. It is important to note that this definition depends on the similarity between A and B. We create pairs of classification tasks A and B by constructing pairs of non-overlapping subsets of the ImageNet dataset.1 These subsets can be chosen to be similar to or different from each other. 

我们已经注意到Gabor滤波器和颜色斑点的奇怪趋势出现在训练自然图像的第一层神经网络中。在本研究中，我们将在任务A上学习的一组特征的一般性程度定义为特征可用于另一个任务B的程度。重要的是要注意该定义取决于A和B之间的相似性。我们通过构造ImageNet数据集的非重叠子集对来创建分类任务A和B对。这些子集可以选择为彼此相似或不同。

To create tasks A and B, we randomly split the 1000 ImageNet classes into two groups each con- taining 500 classes and approximately half of the data, or about 645,000 examples each. We train one eight-layer convolutional network on A and another on B. These networks, which we call baseA and baseB, are shown in the top two rows of Figure 1. We then choose a layer n from {1,2,...,7} and train several new networks. In the following explanation and in Figure 1, we use layer n = 3 as the example layer chosen. First, we define and train the following two networks: 

为了创建任务A和B，我们将1000个ImageNet类随机分成两组，每组包含500个类和大约一半的数据，或者每个大约645,000个示例。我们在A上训练一个八层卷积网络，在B上训练另一个八层卷积网络。这些网络，我们称之为baseA和baseB，显示在图1的前两行。然后我们从{1,2，...中选择一个层n。 。，7}并培训几个新的网络。在以下说明和图1中，我们使用层n = 3作为所选的示例层。首先，我们定义并训练以下两个网络：

- A selffer network B3B: the first 3 layers are copied from baseB and frozen. The five higher layers (4–8) are initialized randomly and trained on dataset B. This network is a control for the next transfer network. (Figure 1, row 3) 
- A transfer network A3B: the first 3 layers are copied from baseA and frozen. The five higher layers (4–8) are initialized randomly and trained toward dataset B. Intuitively, here we copy the first 3 layers from a network trained on dataset A and then learn higher layer features on top of them to classify a new target dataset B. If A3B performs as well as baseB, there is evidence that the third-layer features are general, at least with respect to B. If performance suffers, there is evidence that the third-layer features are specific to A. (Figure 1, row 4) 

* selffer网络B3B：前3个层从baseB复制并冻结。 五个较高层（4-8）随机初始化并在数据集B上训练。该网络是下一个传输网络的控制。 （图1，第3行）
* 传输网络A3B：前3层从baseA复制并冻结。 五个更高层（4-8）被随机初始化并训练到数据集B.直观地，这里我们从在数据集A上训练的网络复制前三个层，然后在它们之上学习更高层特征以对新目标数据集进行分类。 B.如果A3B的性能与baseB相同，则有证据表明第三层特征是通用的，至少就B而言。如果性能受损，则有证据表明第三层特征是A特有的。（图1） ，第4行）

We repeated this process for all n in {1, 2, . . . , 7}2 and in both directions (i.e. AnB and BnA). In the above two networks, the transferred layers are frozen. We also create versions of the above two networks where the transferred layers are fine-tuned: 

我们对{1,2，...中的所有n重复了这个过程。。。 ，7} 2和两个方向（即AnB和BnA）。 在上述两个网络中，传输的层被冻结。 我们还创建了上述两个网络的版本，其中传输的图层经过微调：

* A selffer network B3B+: just like B3B, but where all layers learn.

* A transfer network A3B+: just like A3B, but where all layers learn. 

* 一个selffer网络B3B +：就像B3B一样，但所有层都学习。
* 转移网络A3B +：就像A3B一样，但所有层都学习。

To create base and target datasets that are similar to each other, we randomly assign half of the 1000 ImageNet classes to A and half to B. ImageNet contains clusters of similar classes, particularly dogs and cats, like these 13 classes from the biological family Felidae: {tabby cat, tiger cat, Persian cat, Siamese cat, Egyptian cat, mountain lion, lynx, leopard, snow leopard, jaguar, lion, tiger, cheetah}. On average, A and B will each contain approximately 6 or 7 of these felid classes, meaning that base networks trained on each dataset will have features at all levels that help classify some types of felids. When generalizing to the other dataset, we would expect that the new high-level felid detectors trained on top of old low-level felid detectors would work well. Thus A and B are similar when created by randomly assigning classes to each, and we expect that transferred features will perform better than when A and B are less similar. 

为了创建彼此相似的基础和目标数据集，我们将1000个ImageNet类中的一半随机分配给A，将一半分配给B. ImageNet包含类似类的集群，特别是狗和猫，如生物科Felidae的这13个类：{虎斑猫，虎猫，波斯猫，暹罗猫，埃及猫，山狮，ly ,,豹，雪豹，美洲虎，狮子，虎，猎豹}。平均而言，A和B各自包含大约6或7个这些猫科动物类别，这意味着在每个数据集上训练的基础网络将具有帮助对某些类型的猫科动物进行分类的所有级别的特征。当推广到其他数据集时，我们期望在旧的低级别猫科探测器之上训练的新的高级猫科探测器将很好地工作。因此，A和B在通过随机分配每个类创建时是相似的，并且我们期望传递的特征将比A和B不太相似时表现更好。

Fortunately, in ImageNet we are also provided with a hierarchy of parent classes. This information allowed us to create a special split of the dataset into two halves that are as semantically different from each other as possible: with dataset A containing only man-made entities and B containing natural entities. The split is not quite even, with 551 classes in the man-made group and 449 in the natural group. Further details of this split and the classes in each half are given in the supplementary material. In Section 4.2 we will show that features transfer more poorly (i.e. they are more specific) when the datasets are less similar. 

幸运的是，在ImageNet中，我们还提供了父类的层次结构。这些信息使我们能够将数据集的特殊分割创建为两个尽可能在语义上彼此不同的一半：数据集A仅包含人造实体，B包含自然实体。分裂不是很均匀，人造组有551个班，自然组有449个班。补充材料中给出了这种拆分的更多细节以及每一半的类别。在4.2节中，我们将展示当数据集不太相似时，特征转移更差（即它们更具体）。

![](http://118.25.176.50/md_img/论文_能否迁移1.JPG)

Figure 1: Overview of the experimental treatments and controls. Top two rows: The base networks
are trained using standard supervised backprop on only half of the ImageNet dataset (first row: A
half, second row: B half). The labeled rectangles (e.g. WA1) represent the weight vector learned for
that layer, with the color indicating which dataset the layer was originally trained on. The vertical,
ellipsoidal bars between weight vectors represent the activations of the network at each layer. Third
row: In the selffer network control, the first n weight layers of the network (in this example, n = 3)
are copied from a base network (e.g. one trained on dataset B), the upper 8 − n layers are randomly
initialized, and then the entire network is trained on that same dataset (in this example, dataset B).
The first n layers are either locked during training (“frozen” selffer treatment B3B) or allowed to
learn (“fine-tuned” selffer treatment B3B+). This treatment reveals the occurrence of fragile co-
adaptation, when neurons on neighboring layers co-adapt during training in such a way that cannot
be rediscovered when one layer is frozen. Fourth row: The transfer network experimental treatment
is the same as the selffer treatment, except that the first n layers are copied from a network trained
on one dataset (e.g. A) and then the entire network is trained on the other dataset (e.g. B). This
treatment tests the extent to which the features on layer n are general or specific.

图1：实验处理和对照概述。前两行：基础网络仅使用标准监督背板在ImageNet数据集的一半上进行训练（第一行：一半，第二行：B半）。标记的矩形（例如WA1）表示为该层学习的权重向量，其颜色指示该层最初被训练的数据集。权重向量之间的垂直椭球柱表示每层网络的激活。第三行：在selffer网络控制中，从基础网络（例如，在数据集B上训练的一个）复制网络的前n个权重层（在该示例中，n = 3），上部8-n层被随机初始化，然后整个网络在同一数据集上进行训练（在本例中为数据集B）。前n个层在训练期间被锁定（“冷冻”selffer治疗B3B）或允许学习（“微调”selffer治疗B3B +）。这种治疗揭示了脆弱的共适应的发生，当相邻层上的神经元在训练期间以这样的方式共同适应时，当一层被冻结时不能被重新发现。第四行：转移网络实验处理与selffer处理相同，不同之处在于前一个n层是从一个数据集（例如A）上训练的网络复制的，然后整个网络在另一个数据集上训练（例如B） 。该处理测试n层上的特征是一般的或特定的程度。

##Experimental Setup实验设置

Since Krizhevsky et al. (2012) won the ImageNet 2012 competition, there has been much interest
and work toward tweaking hyperparameters of large convolutional models. However, in this study
we aim not to maximize absolute performance, but rather to study transfer results on a well-known
architecture. We use the reference implementation provided by Caffe (Jia et al., 2014) so that our
results will be comparable, extensible, and useful to a large number of researchers. Further details of
the training setup (learning rates, etc.) are given in the supplementary material, and code and param-
eter files to reproduce these experiments are available at http://yosinski.com/transfer.

自Krizhevsky等人。 （2012）赢得了ImageNet 2012竞赛，人们对调整大型卷积模型的超参数有很大的兴趣和努力。 然而，在本研究中，我们的目标不是最大化绝对性能，而是研究在众所周知的体系结构上的转移结果。 我们使用Caffe（Jia et al。，2014）提供的参考实现，以便我们的结果与大量研究人员具有可比性，可扩展性和实用性。 有关培训设置（学习率等）的更多详细信息，请参阅补充材料，有关重现这些实验的代码和参数文件，请访问http://yosinski.com/transfer。

## Results and Discussion 结果和讨论

We performed three sets of experiments. The main experiment has random A/B splits and is dis-
cussed in Section 4.1. Section 4.2 presents an experiment with the man-made/natural split. Sec-
tion 4.3 describes an experiment with random weights.

我们进行了三组实验。 主要实验有随机A / B分裂，将在4.1节中讨论。 第4.2节介绍了人造/自然分裂的实验。 第4.3节描述了一个随机权重的实验。

![](http://118.25.176.50/md_img/论文_能否迁移2.JPG)

![](http://118.25.176.50/md_img/论文_能否迁移3.JPG)

Figure 2: The results from this paper’s main experiment. Top: Each marker in the figure represents
the average accuracy over the validation set for a trained network. The white circles above n =
0 represent the accuracy of baseB. There are eight points, because we tested on four separate
random A/B splits. Each dark blue dot represents a BnB network. Light blue points represent
BnB+ networks, or fine-tuned versions of BnB. Dark red diamonds are AnB networks, and light
red diamonds are the fine-tuned AnB+ versions. Points are shifted slightly left or right for visual
clarity. Bottom: Lines connecting the means of each treatment. Numbered descriptions above each
line refer to which interpretation from Section 4.1 applies.

图2：本文主要实验的结果。 上图：图中的每个标记代表经过训练的网络的验证集的平均准确度。 n = 0以上的白圈代表baseB的准确性。 有八点，因为我们测试了四个独立的随机A / B分割。 每个深蓝色点代表BnB网络。 浅蓝色点表示BnB +网络或BnB的微调版本。 暗红色钻石是AnB网络，浅红色钻石是经过微调的AnB +版本。 为了清晰起见，点向左或向右略微移动。 底部：连接每种处理方式的线。 每行上方的编号描述指的是4.1节中的哪种解释适用。

### Similar Datasets: Random A/B splits 类似数据集：随机A / B分割

The results of all A/B transfer learning experiments on randomly split (i.e. similar) datasets are
shown3 in Figure 2. The results yield many different conclusions. In each of the following interpre-
tations, we compare the performance to the base case (white circles and dotted line in Figure 2).

随机分割（即相似）数据集的所有A / B转移学习实验的结果如图2所示。结果得出许多不同的结论。 在以下每个解释中，我们将性能与基本情况进行比较（图2中的白色圆圈和虚线）。

1. The white baseB circles show that a network trained to classify a random subset of 500 classes attains a top-1 accuracy of 0.625, or 37.5% error. This error is lower than the 42.5% top-1 error attained on the 1000-class network. While error might have been higher because the network is trained on only half of the data, which could lead to more overfitting, the net result is that error is lower because there are only 500 classes, so there are only half as many ways to make mistakes. 

2. The dark blue BnB points show a curious behavior. As expected, performance at layer one is the same as the baseB points. That is, if we learn eight layers of features, save the first layer of learned Gabor features and color blobs, reinitialize the whole network, and retrain it toward the same task, it does just as well. This result also holds true for layer 2. However, layers 3, 4, 5, and 6, particularly 4 and 5, exhibit worse performance. This performance drop is evidence that the original network contained fragile co-adapted features on successive layers, that is, features that interact with each other in a complex or fragile way such that this co-adaptation could not be relearned by the upper layers alone. Gradient descent was able to find a good solution the first time, but this was only possible because the layers were jointly trained. By layer 6 performance is nearly back to the base level, as is layer 7. As we get closer and closer to the final, 500-way softmax output layer 8, there is less to relearn, and apparently relearning these one or two layers is simple enough for gradient descent to find a good solution. Alternately, we may say that there is less co-adaptation of features between layers 6 & 7 and between 7 & 8 than between previous layers. To our knowledge it has not been previously observed in the literature that such optimization difficulties may be worse in the middle of a network than near the bottom or top. 

3. The light blue BnB+ points show that when the copied, lower-layer features also learn on the target dataset (which here is the same as the base dataset), performance is similar to the base case. Such fine-tuning thus prevents the performance drop observed in the BnB networks. 

4. The dark red AnB diamonds show the effect we set out to measure in the first place: the transfer- ability of features from one network to another at each layer. Layers one and two transfer almost perfectly from A to B, giving evidence that, at least for these two tasks, not only are the first-layer Gabor and color blob features general, but the second layer features are general as well. Layer three shows a slight drop, and layers 4-7 show a more significant drop in performance. Thanks to the BnB points, we can tell that this drop is from a combination of two separate effects: the drop from lost co-adaptation and the drop from features that are less and less general. On layers 3, 4, and 5, the first effect dominates, whereas on layers 6 and 7 the first effect diminishes and the specificity of representation dominates the drop in performance. 

   Although examples of successful feature transfer have been reported elsewhere in the literature (Girshick et al., 2013; Donahue et al., 2013b), to our knowledge these results have been limited to noticing that transfer from a given layer is much better than the alternative of training strictly on the target task, i.e. noticing that the AnB points at some layer are much better than training all layers from scratch. We believe this is the first time that (1) the extent to which transfer is successful has been carefully quantified layer by layer, and (2) that these two separate effects have been decoupled, showing that each effect dominates in part of the regime. 

5. The light red AnB+ diamonds show a particularly surprising effect: that transferring features and then fine-tuning them results in networks that generalize better than those trained directly on the target dataset. Previously, the reason one might want to transfer learned features is to enable training without overfitting on small target datasets, but this new result suggests that transferring features will boost generalization performance even if the target dataset is large. Note that this effect should not be attributed to the longer total training time (450k base iterations + 450k fine- tuned iterations for AnB+ vs. 450k for baseB), because the BnB+ networks are also trained for the same longer length of time and do not exhibit this same performance improvement. Thus, a plausible explanation is that even after 450k iterations of fine-tuning (beginning with completely random top layers), the effects of having seen the base dataset still linger, boosting generalization performance. It is surprising that this effect lingers through so much retraining. This generalization improvement seems not to depend much on how much of the first network we keep to initialize the second network: keeping anywhere from one to seven layers produces improved performance, with slightly better performance as we keep more layers. The average boost across layers 1 to 7 is 1.6% over the base case, and the average if we keep at least five layers is 2.1%.4 The degree of performance boost is shown in Table 1. 

1. 白色baseB圆圈表明，经过训练以对500个类别的随机子集进行分类的网络获得0.625的前1准确度，或37.5％的误差。此错误低于1000级网络上达到的42.5％top-1错误。虽然错误可能更高，因为网络只训练了一半的数据，这可能导致更多的过度拟合，最终的结果是错误较低，因为只有500个类，所以只有一半的方法可以做到错误。
2. 深蓝色的BnB点显示出一种奇怪的行为。正如所料，第一层的表现与baseB点相同。也就是说，如果我们学习八层特征，保存第一层学习的Gabor特征和颜色斑点，重新初始化整个网络，并将其重新训练为同一任务，它也同样如此。该结果也适用于层2.然而，层3,4,5和6，特别是4和5，表现出更差的性能。这种性能下降证明了原始网络在连续层上包含易碎的共适应特征，即，以复杂或脆弱的方式彼此交互的特征，使得这种共同适应不能仅由上层重新学习。梯度下降第一次能够找到一个好的解决方案，但这只是可能的，因为这些层是共同训练的。通过第6层，性能几乎回到了基础层，就像第7层一样。随着我们越来越接近最终的500路softmax输出层8，重新学习的次数越来越少，显然重新学习这一层或两层是足够简单的梯度下降找到一个很好的解决方案。或者，我们可以说，层6和7之间以及7和8之间的特征的共同适应性小于先前层之间的特征的共同适应性。据我们所知，以前在文献中没有观察到这种优化困难在网络中间可能比在底部或顶部附近更差。
3. 浅蓝色BnB +点表明，当复制的低层特征也学习目标数据集（这里与基础数据集相同）时，性能类似于基本情况。因此，这种微调可以防止在BnB网络中观察到的性能下降。
4. 暗红色的AnB钻石首先展示了我们开始测量的效果：每层的特征从一个网络到另一个网络的传递能力。第一层和第二层几乎完美地从A转移到B，证明至少对于这两个任务，不仅第一层Gabor和颜色blob特征是通用的，而且第二层特征也是通用的。第三层显示轻微下降，第4-7层显示性能下降更明显。由于BnB点，我们可以看出这种下降来自两个独立效果的组合：失去共同适应的下降和来自越来越不普遍的特征的下降。在第3,4和5层，第一个效应占主导地位，而在第6和第7层，第一个效应减弱，表现的特异性主导性能下降。
   尽管已经在文献的其他地方报道了成功特征转移的例子（Girshick等，2013; Donahue等，2013b），但据我们所知，这些结果仅限于注意到从给定层的转移比严格训练目标任务的替代方法，即注意某层的AnB点比从头开始训练所有层要好得多。我们认为这是第一次（1）转移成功的程度已经逐层仔细量化，以及（2）这两个单独的效应已经解耦，表明每个效应在部分政权中占主导地位。
5. 浅红色AnB +钻石显示出特别令人惊讶的效果：传递特征然后对它们进行微调会导致网络比直接在目标数据集上训练的网络更好地推广。以前，人们可能想要转移学习特征的原因是为了在没有过度拟合小目标数据集的情况下启用训练，但是这个新结果表明，即使目标数据集很大，传递特征也会提高泛化性能。请注意，这种影响不应归因于较长的总训练时间（对于AnB +为450k基本迭代+ 450k微调迭代而对于baseB为450k），因为BnB +网络也经历了相同更长时间的训练而没有表现出同样的性能提升。因此，一个似乎合理的解释是，即使在450k次微调之后（从完全随机的顶层开始），看到基础数据集的效果仍然存在，从而提高了泛化性能。令人惊讶的是，这种影响在如此多的再培训中徘徊不前。这种泛化改进似乎并不太依赖于我们保留初始化第二个网络的第一个网络的多少：保持一到七层的任何地方都会产生改进的性能，并且随着我们保留更多层，性能稍好一些。第1层到第7层的平均增强率比基础情况高1.6％，如果我们保持至少5层，平均值增加2.1％.4性能提升程度如表1所示。

![](http://118.25.176.50/md_img/论文_能否迁移4.JPG)

### Dissimilar Datasets: Splitting Man-made and Natural Classes Into Separate Datasets 不同的数据集：将人造和自然类拆分为单独的数据集

As mentioned previously, the effectiveness of feature transfer is expected to decline as the base and target tasks become less similar. We test this hypothesis by comparing transfer performance on similar datasets (the random A/B splits discussed above) to that on dissimilar datasets, created by assigning man-made object classes to A and natural object classes to B. This man-made/natural split creates datasets as dissimilar as possible within the ImageNet dataset. 

The upper-left subplot of Figure 3 shows the accuracy of a baseA and baseB network (white circles) and BnA and AnB networks (orange hexagons). Lines join common target tasks. The upper of the two lines contains those networks trained toward the target task containing natural categories (baseB and AnB). These networks perform better than those trained toward the man-made categories, which may be due to having only 449 classes instead of 551, or simply being an easier task, or both. 

如前所述，随着基础和目标任务变得不那么相似，特征转移的有效性预计会下降。 我们通过比较类似数据集（上面讨论的随机A / B分裂）与不同数据集上的传递性能来测试这一假设，通过将人造对象类分配给A和自然对象类创建为B.这种人造/自然 split在ImageNet数据集中创建尽可能不相似的数据集。
图3的左上方子图显示了baseA和baseB网络（白色圆圈）和BnA和AnB网络（橙色六边形）的准确性。 行连接常见的目标任务。 两条线的上部包含那些训练朝向包含自然类别（baseB和AnB）的目标任务的网络。 这些网络的性能优于那些受过人工训练的人群，这可能是由于只有449个班级而不是551个，或者只是一个更容易完成的任务，或者两者兼而有之。

### Random Weights 随机权重

We also compare to random, untrained weights because Jarrett et al. (2009) showed — quite strik- ingly — that the combination of random convolutional filters, rectification, pooling, and local nor- malization can work almost as well as learned features. They reported this result on relatively small networks of two or three learned layers and on the smaller Caltech-101 dataset (Fei-Fei et al., 2004). It is natural to ask whether or not the nearly optimal performance of random filters they report carries over to a deeper network trained on a larger dataset. 

我们还比较了随机的，未经训练的重量，因为Jarrett等人。 （2009）非常严格地表明，随机卷积滤波器，校正，汇集和局部正规化的组合几乎与学习的特征一样有效。他们在两个或三个学习层和较小的Caltech-101数据集的相对较小的网络上报告了这一结果（Fei-Fei等，2004）。很自然地会问，他们报告的随机滤波器的近乎最佳性能是否会延伸到更大数据集上训练的更深层网络。

The upper-right subplot of Figure 3 shows the accuracy obtained when using random filters for the first n layers for various choices of n. Performance falls off quickly in layers 1 and 2, and then drops to near-chance levels for layers 3+, which suggests that getting random weights to work in convolutional neural networks may not be as straightforward as it was for the smaller network size and smaller dataset used by Jarrett et al. (2009). However, the comparison is not straightforward. Whereas our networks have max pooling and local normalization on layers 1 and 2, just as Jarrett et al. (2009) did, we use a different nonlinearity (relu(x) instead of abs(tanh(x))), different layer sizes and number of layers, as well as other differences. Additionally, their experiment only consid- ered two layers of random weights. The hyperparameter and architectural choices of our network collectively provide one new datapoint, but it may well be possible to tweak layer sizes and random initialization details to enable much better performance for random weights.5 

图3的右上子图显示了对于n的各种选择使用前n个层的随机滤波器时获得的精度。在第1层和第2层，性能迅速下降，然后在第3层和第3层下降到接近机会的水平，这表明在卷积神经网络中使用随机权重可能不像在较小的网络规模和更小的网络中那样简单Jarrett等人使用的数据集。 （2009年）。但是，比较并不简单。而我们的网络在第1层和第2层都有最大池和局部归一化，就像Jarrett等人一样。 （2009），我们使用不同的非线性（relu（x）代替abs（tanh（x））），不同的层大小和层数，以及其他差异。此外，他们的实验只考虑了两层随机权重。我们网络的超参数和架构选择共同提供了一个新的数据点，但很可能调整层大小和随机初始化细节，以便为随机权重提供更好的性能.5

The bottom subplot of Figure 3 shows the results of the experiments of the previous two sections
after subtracting the performance of their individual base cases. These normalized performances
are plotted across the number of layers n that are either random or were trained on a different,
base dataset. This comparison makes two things apparent. First, the transferability gap when using
frozen features grows more quickly as n increases for dissimilar tasks (hexagons) than similar tasks
(diamonds), with a drop by the final layer for similar tasks of only 8% vs. 25% for dissimilar tasks.
Second, transferring even from a distant task is better than using random filters. One possible reason
this latter result may differ from Jarrett et al. (2009) is because their fully-trained (non-random)
networks were overfitting more on the smaller Caltech-101 dataset than ours on the larger ImageNet dataset, making their random filters perform better by comparison. In the supplementary material,
we provide an extra experiment indicating the extent to which our networks are overfit.

图3的底部子图显示了在减去各个基本情况的性能之后前两个部分的实验结果。这些归一化的性能在层数n上绘制，这些层是随机的或在不同的基础数据集上训练。这种比较使两件事显而易见。首先，使用冻结特征时的可转移性差距随着不同任务（六边形）的n增加而增长得比同类任务（钻石）更快，最终层对于类似任务的下降仅为8％而不同任务为25％。其次，即使从远程任务转移也比使用随机过滤器更好。后者的一个可能原因可能与Jarrett等人有所不同。 （2009年）是因为他们完全训练的（非随机）网络在较大的ImageNet数据集上比较小的Caltech 101数据集更多地过度拟合，使得它们的随机过滤器相比表现更好。在补充材料中，我们提供了一个额外的实验，表明我们的网络过度适应的程度。

![](http://118.25.176.50/md_img/论文_能否迁移5.JPG)

## Conclusions结论

We have demonstrated a method for quantifying the transferability of features from each layer of
a neural network, which reveals their generality or specificity. We showed how transferability is
negatively affected by two distinct issues: optimization difficulties related to splitting networks in
the middle of fragilely co-adapted layers and the specialization of higher layer features to the original
task at the expense of performance on the target task. We observed that either of these two issues
may dominate, depending on whether features are transferred from the bottom, middle, or top of
the network. We also quantified how the transferability gap grows as the distance between tasks
increases, particularly when transferring higher layers, but found that even features transferred from
distant tasks are better than random weights. Finally, we found that initializing with transferred
features can improve generalization performance even after substantial fine-tuning on a new task,
which could be a generally useful technique for improving deep neural network performance.

我们已经证明了一种量化神经网络每层特征可转移性的方法，揭示了它们的一般性或特异性。我们展示了可转移性如何受到两个不同问题的负面影响：与脆弱共同适应层中间分裂网络相关的优化困难以及高层特征与原始任务的专业化，而牺牲了目标任务的性能。我们观察到这两个问题中的任何一个都可能占主导地位，这取决于功能是从网络的底部，中间还是顶部传输。我们还量化了可转移性差距如何随着任务之间的距离增加而增加，特别是在传输更高层时，但发现即使从远程任务转移的特征也优于随机权重。最后，我们发现，即使在对新任务进行大量微调之后，使用传输特征进行初始化也可以提高泛化性能，这可能是提高深度神经网络性能的一种常用技术。





