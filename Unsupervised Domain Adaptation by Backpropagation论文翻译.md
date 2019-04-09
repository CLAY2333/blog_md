---
title: Unsupervised Domain Adaptation by Backpropagation论文翻译
date: 2018-11-12 15:39:53
categories: 论文翻译
tags: [论文,机器学习,深度学习,迁移学习]
---

<Excerpt in index | 首页摘要> 

深度迁移学习经典论文

<!-- more -->

<The rest of contents | 余下全文>



# Unsupervised Domain Adaptation by Backpropagation

## 摘要

Top-performing deep architectures are trained on mas- sive amounts of labeled data. In the absence of labeled data for a certain task, domain adaptation often provides an at- tractive option given that labeled data of similar nature but from a different domain (e.g. synthetic images) are avail- able. Here, we propose a new approach to domain adap- tation in deep architectures that can be trained on large amount of labeled data from the source domain and large amount of unlabeled data from the target domain (no la- beled target-domain data is necessary).
As the training progresses, the approach promotes the emergence of “deep” features that are (i) discriminative for the main learning task on the source domain and (ii) invari- ant with respect to the shift between the domains. We show that this adaptation behaviour can be achieved in almost any feed-forward model by augmenting it with few standard layers and a simple new gradient reversal layer. The result- ing augmented architecture can be trained using standard backpropagation. Overall the whole approach can be im- plemented with little effort using any of the deep-learning packages.

**表现最佳的深层架构接受大量标记数据的培训。在没有针对特定任务的标记数据的情况下，域适应通常提供吸引选项，因为具有相似性质但来自不同域（例如合成图像）的标记数据是可用的。在这里，我们提出了一种深层体系结构中域适应的新方法，可以对来自源域的大量标记数据和来自目标域的大量未标记数据进行训练（无需标记目标域数据） ）。**
**随着培训的进行，该方法促进了“深层”特征的出现，这些特征是（i）对源域上的主要学习任务进行区分，以及（ii）关于域之间转换的不变性。我们表明，通过增加几个标准层和一个简单的新梯度反转层，几乎可以在任何前馈模型中实现这种适应行为。可以使用标准反向传播来训练结果增强的体系结构。总的来说，使用任何深度学习软件包都可以轻松实现整个方法。**

## 引言

Deep feed-forward architectures have brought impres- sive advances to the state-of-the-art across a wide variety tasks within computer vision and beyond. At the moment, however, these leaps in performance emerge only when a large amount of labeled training data is available. At the same time, for problems lacking labeled data, it may be still possible to obtain training sets that are big enough for train- ing large-scale deep models, but that suffer from the shift in data distribution from the actual data encountered at “test time”. One particularly important example is synthetic or semi-synthetic imagery, which may come in abundance and fully labeled, but which inevitably look different from real data [13, 20, 23, 21].
Learning a discriminative classifier or other predictor in the presence of a shift between training and testing distri- butions is known as domain adaptation (DA). A number of approaches to domain adaptation has been suggested in the context of shallow learning, e.g. in the situation when data representation/features are given and fixed. The proposed approaches then build the mappings between the source (training-time) and the target (test-time) domains, so that the classifier learned for the source domain can also be ap- plied to the target domain, when composed with the learned mapping between domains. The appeal of the domain adap- tation approaches is the ability to learn a mapping between domains in the situation when the target domain data are either fully unlabeled (unsupervised domain annotation) or have few labeled samples (semi-supervised domain adap- tation). Below, we focus on the harder unsupervised case, although the proposed approach can be generalized to the semi-supervised case rather straightforwardly.

**深度前馈架构为计算机视觉及其他领域的各种任务带来了最先进的技术进步。然而，目前，只有当有大量标记的训练数据可用时，才会出现这些性能上的飞跃。同时，对于缺少标记数据的问题，仍然可能获得足够大的训练集来训练大规模的深度模型，但是数据分布的变化受到“实际数据”的影响。考试时间“。一个特别重要的例子是合成或半合成图像，它可能丰富且完全标记，但不可避免地看起来与实际数据不同[13,20,23,21]。**
**在训练和测试分布之间存在转换时学习判别分类器或其他预测器称为域适应（DA）。在浅层学习的背景下，已经提出了许多领域适应的方法，例如：在给出和修复数据表示/特征的情况下。然后，所提出的方法构建源（训练时间）和目标（测试时间）域之间的映射，以便为源域学习的分类器也可以应用于目标域。域之间的映射。域适应方法的吸引力在于，当目标域数据完全未标记（无监督域注释）或具有少量标记样本（半监督域适应）时，能够学习域之间的映射。下面，我们关注更难的无监督案例，尽管提出的方法可以直接推广到半监督案例。**

Unlike most papers previous papers on domain adap- tation that worked with fixed feature representations, we focus on combining domain adaptation and deep feature learning within one training process (deep domain adap- tation). Our goal is to embed domain adaptation into the process of learning representation, so that the final classi- fication decisions are made based on features that are both discriminative and invariant to the change of domains, i.e. have the same or very similar distributions in the source and the target domains. In this way, the obtained feed-forward network can be applicable to the target domain without be- ing hindered by the shift between the two domains.
We thus focus on learning features that combine (i) discriminativeness and (ii) domain-invariance. This is achieved by jointly optimizing the underlying features as well as two discriminative classifiers operating on these fea- tures: (i) the label predictor that predicts class labels and is used at test time and (ii) the domain classifier that discrim- inates between the source and the target domains during training. While the parameters of the classifiers are opti- mized in order to minimize their error on the training set, the parameters of the underlying deep feature mapping are optimized in order to minimize the loss of the label classi- fier and to maximize the loss of the domain classifier. The latter encourages domain-invariant features to emerge in the course of the optimization.

**与大多数关于适用于固定特征表示的领域适应的论文不同，我们专注于在一个训练过程（深度域自适应）中结合领域适应和深度特征学习。我们的目标是将域自适应嵌入到学习表示的过程中，以便最终的分类决策基于对域的变化具有判别性和不变性的特征，即在源中具有相同或非常相似的分布。和目标域。通过这种方式，所获得的前馈网络可以适用于目标域，而不受两个域之间的转换的阻碍。**
**因此，我们专注于学习结合（i）判别性和（ii）域不变性的特征。这是通过联合优化基础特征以及在这些特征上运行的两个判别分类器来实现的：（i）预测类标签并在测试时使用的标签预测器和（ii）区分它们的域分类器培训期间的源和目标域。虽然优化分类器的参数以最小化它们在训练集上的误差，但是底层深度特征映射的参数被优化以便最小化标签分类器的损失并最大化丢失。域分类器。后者鼓励在优化过程中出现域不变特征。**

Crucially, we show that all three training processes can be embedded into an appropriately composed deep feed- forward network (Figure 1) that uses standard layers and loss functions, and can be trained using standard backprop- agation algorithms based on stochastic gradient descent or its modifications (e.g. momentum). Our approach is generic as it can be used to add domain adaptation to any existing feed-forward architecture that is trainable by backpropaga- tion. In practice, the only non-standard component of the proposed architecture is a rather trivial gradient reversal layer that leaves the input unchanged during forward propa- gation and reverses the gradient by multiplying it by a neg- ative scalar during the backpropagation. 

Below, we detail the proposed approach to domain adap- tation in deep architectures, and present prelimenary results on traditional deep learning datasets (such as MNIST [12] and SVHN [14]) that clearly demonstrate the unsupervised domain adaptation ability of the proposed method. 

**至关重要的是，我们表明所有三个训练过程都可以嵌入到使用标准层和损失函数的适当组合的深度前馈网络（图1）中，并且可以使用基于随机梯度下降或其随机梯度下降的标准反向传播算法进行训练。修改（例如动量）。我们的方法是通用的，因为它可以用于将域适配添加到可通过反向传播训练的任何现有前馈架构。在实践中，所提出的架构中唯一的非标准组件是相当简单的梯度反转层，其在前向传播期间使输入保持不变，并且通过在反向传播期间将其乘以负标量来反转梯度。**
**下面，我们详细介绍了深层架构中域适应的建议方法，并提出了传统深度学习数据集（如MNIST [12]和SVHN [14]）的初步结果，清楚地证明了该方法的无监督域适应能力。** 

## 相关工作

A large number of domain adaptation methods have been proposed over the recent years, and here we focus on the most related ones. Multiple methods perform unsupervised domain adaptation by matching the feature distributions in the source and the target domains. Some approaches perform this by reweighing or selecting samples from the source domain [3, 11, 7], while others seek an explicit fea- ture space transformation that would map source distribu- tion into the target ones [16, 10, 2]. An important as- pect of the distribution matching approach is the way the (dis)similarity between distributions is measured. Here, one popular choice is matching the distribution means in the kernel-reproducing Hilbert space [3, 11], whereas [8, 5] map the principal axes associated with each of the distri- butions. Our approach also attempts to match feature space distributions, however this is accomplished by modifying the feature representation itself rather than by reweighing or geometric transformation. Also, our method uses (im- plicitly) a rather different way to measure the disparity be- tween distributions based on their separability by a deep discriminatively-trained classifier.

**近年来已经提出了大量的域自适应方法，这里我们关注最相关的方法。多种方法通过匹配源域和目标域中的特征分布来执行无监督的域自适应。一些方法通过重新加权或从源域中选择样本来实现这一点[3,11,7]，而其他方法则寻求一种明确的特征空间变换，将源分布映射到目标分布[16,10,2]。分布匹配方法的一个重要方面是测量分布之间（dis）相似性的方式。这里，一个流行的选择是匹配内核再现希尔伯特空间[3,11]中的分布均值，而[8,5]则映射与每个分布相关的主轴。我们的方法也尝试匹配特征空间分布，但这是通过修改特征表示本身而不是通过重新加权或几何变换来实现的。此外，我们的方法使用（明确地）一种相当不同的方式来测量分布之间的差异，这是基于它们通过深度判别训练的分类器的可分离性。**

Several approaches perform gradual transition from the source to the target domain [10, 8] by a gradual change of the training distribution. Among these methods, [17] does this in a “deep” way by the layerwise training of a sequence of deep autoencoders, while gradually replacing source-domain samples with target-domain samples. This improves over a similar approach of [6] that simply trains a single deep autoencoder for both domains. In both approaches, the actual classifier/predictor is learned in a separate step using the feature representation learned by autoencoder(s). In contrast to [6, 17], our approach performs feature learning, domain adaptation and classifier learning jointly, in a unified architecture, and using a single learning algorithm (backpropagation). We therefore argue that our approach is much simpler (both conceptually and in terms of its implementation).
While the above approaches perform unsupervised domain adaptation, there are approaches that perform supervised domain adaptation by exploiting labeled data from the target domain. In the context of deep feed-forward architectures, such data can be used to “fine-tune” the network trained on the source domain [24, 15, 1]. Our approach does not require labeled target-domain data. At the same time, it can easily incorporate such data when they are available.
The work that is most related to ours is the recent (and concurrent) technical report [9] on adversarial networks. While their goal is quite different (building generative deep networks that can synthesize samples), the way they measure and minimize the discrepancy between the distribution of the training data and the distribution of the synthesized data is very similar to the way our architecture measures and minimizes the discrepancy between feature distributions for the two domains.

**通过逐渐改变训练分布，几种方法逐渐从源域转移到目标域[10,8]。在这些方法中，[17]通过深层自动编码器序列的分层训练以“深度”方式进行，同时逐渐用目标域样本替换源域样本。这改进了[6]的类似方法，简单地为两个域训练单个深度自动编码器。在两种方法中，使用由自动编码器学习的特征表示在单独的步骤中学习实际的分类器/预测器。与[6,17]相比，我们的方法在统一的体系结构中使用单一学习算法（反向传播）联合执行特征学习，域自适应和分类器学习。因此，我们认为我们的方法更简单（在概念上和在实施方面）。
虽然上述方法执行无监督域自适应，但是存在通过利用来自目标域的标记数据来执行监督域自适应的方法。在深度前馈体系结构的背景下，这些数据可用于“微调”在源域[24,15,1]上训练的网络。我们的方法不需要标记的目标域数据。同时，它可以在可用时轻松合并这些数据。
与我们最相关的工作是关于对抗性网络的最新（并发）技术报告[9]。虽然他们的目标是完全不同的（构建可以合成样本的生成深度网络），但他们测量和最小化训练数据分布与合成数据分布之间的差异的方式与我们的架构测量和最小化的方式非常相似两个域的特征分布之间的差异。**

## Deep Domain Adaptation深度域适应

### 模型 

We now detail the proposed model for the domain adap- tation. We assume that the model works with input sam- ples x ∈ X, where X is some input space (e.g. images of a certain size) and certain labels (output) y from the label space Y . Below, we assume classification problems where Y is a finite set (Y = {1,2,...L}), however our approach is generic and can handle any output label space that other deep feed-forward models can handle. We further assume that there exist two distributions S(x, y) and T (x, y) on X ⊗ Y , which will be referred to as the source distribu- tion and the target distribution (or the source domain and the target domain). Both distributions are assumed com- plex and unknown, and furthermore similar but different (in other words, S is “shifted” from T by some domain shift). 

**我们现在详细介绍了域适应的建议模型。 我们假设模型使用输入样本x∈X，其中X是一些输入空间（例如，某个大小的图像）和来自标签空间Y的某些标签（输出）y。 下面，我们假设分类问题，其中Y是有限集（Y = {1,2，... L}），但是我们的方法是通用的，可以处理其他深度前馈模型可以处理的任何输出标签空间。 我们进一步假设在X⊗Y上存在两个分布S（x，y）和T（x，y），它们将被称为源分布和目标分布（或源域和目标域））。 两种分布都假设复杂且未知，而且相似但不同（换句话说，S通过某种域移位从T“移位”）。**

Our ultimate goal is to be able to predict labels y given the input x for the target distribution. At training time, we have an access to a large set of training samples {x1, x2, . . . , xN } from both the source and the target do- mains distributed according to the marginal distributions S(x) and T (x). We denote with di the binary variable (do- main label) for the ith example, which indicates whether xi come from the source distribution (xi ∼S (x) if di =0) or from the target distribution (xi ∼T (x) if di =1). For the ex- amples from the source distribution (di=0) the correspond- ing labels yi ∈ Y are known at training time. For the ex- amples from the target domains, we do not know the labels at training time, and we want to predict such labels at test time.

**我们的最终目标是能够在给定目标分布的输入x的情况下预测标签y。 在训练时，我们可以访问大量训练样本{x1，x2 ,.。。 来自源和目标的xN}根据边际分布S（x）和T（x）分布。 我们用di表示第i个例子的二元变量（do-main标签），它表示xi是来自源分布（xi~S（x），如果di = 0）或来自目标分布（xi~T（x） ）如果di = 1）。 对于源分布（di = 0）的例子，相应的标记yi∈Y在训练时是已知的。 对于目标域的示例，我们在训练时不知道标签，我们希望在测试时预测这些标签。**

![](http://118.25.176.50/md_img/论文_深度迁移适应1.JPG)

Figure 1.The proposed architecture includes a deep feature extractor (green) and a deep label predictor (blue), which together form a standard feed-forward architecture. Unsupervised domain adaptation is achieved by adding a domain classifier (red) connected to the feature extractor via a gradient reversal layer that multiplies the gradient by a certain negative constant during the backpropagation-based training. Otherwise, the training proceeds in a standard way and minimizes the label prediction loss (for source examples) and the domain classification loss (for all samples). Gradient reversal ensures that the feature distributions over the two domains are made similar (as indistinguishable as possible for the domain classifier), thus resulting in the domain-invariant features.

**图1.所提出的架构包括深度特征提取器（绿色）和深度标签预测器（蓝色），它们共同构成标准的前馈架构。 通过在基于反向传播的训练期间将梯度乘以某个负常数的梯度反转层添加连接到特征提取器的域分类器（红色）来实现无监督域自适应。 否则，培训以标准方式进行，并最小化标签预测损失（对于源示例）和域分类丢失（对于所有样本）。 梯度反转确保两个域上的特征分布相似（对于域分类器尽可能无法区分），从而产生域不变特征。**

We now define a deep feed-forward architecture that for each input x predicts its label y ∈ Y and its domain label d ∈ {0, 1}. We decompose such mapping into three parts. We assume that the input x is first mapped by a mapping Gf (a feature extractor) to a D-dimensional feature vector f ∈ RD. The feature mapping may also include several feed-forward layers and we denote the vector of parame- ters of all layers in this mapping as θf, i.e. f = Gf(x;θf). Then, the feature vector f is mapped by a mapping Gy (la- bel predictor) to the label y, and we denote the parameters of this mapping with θy. Finally, the same feature vector f is mapped to the domain label d by a mapping Gd (domain classifier) with the parameters θd (Figure 1). 

During the learning stage, we aim to minimize the label prediction loss on the annotated part (i.e. the source part) of the training set, and the parameters of both the feature ex- tractor and the label predictor are thus optimized in order to minimize the empirical loss for the source domain samples. This ensures the discriminativeness of the features f and the overall good prediction performance of the combination of the feature extractor and the label predictor on the source domain. 

**我们现在定义一个深度前馈架构，对于每个输入x，它预测其标签y∈Y及其域标签d∈{0,1}。我们将这种映射分解为三个部分。我们假设输入x首先由映射Gf（特征提取器）映射到D维特征向量f∈RD。特征映射还可以包括几个前馈层，并且我们将该映射中的所有层的参数矢量表示为θf，即f = Gf（x;θf）。然后，通过映射Gy（标签预测器）将特征向量f映射到标签y，并且我们用θy表示该映射的参数。最后，通过具有参数θd的映射Gd（域分类器）将相同的特征向量f映射到域标签d（图1）。
在学习阶段，我们的目标是最小化训练集的注释部分（即源部分）上的标签预测损失，并且因此优化特征提取器和标签预测器的参数以便最小化源域样本的经验损失。这确保了特征f的辨别力以及特征提取器和源域上的标签预测器的组合的总体良好预测性能。**

At the same time, we want to make the features f domain-invariant. That is, we want to make the distributions S(f) = {Gf(x;θf)|x∼S(x)} and T(f) = {Gf (x; θf ) | x∼T (x)} to be similar. Under the covariate shift assumption, this would make the label prediction accuracy on the target domain to be the same as on the source domain [18]. Measuring the dissimilarity of the distributions S(f) and T(f) is however non-trivial, given that f is high-dimensional, and that the distributions themselves are constantly changing as learning progresses. One way to estimate the dissimilarity is to look at the loss of the domain classifier Gd, provided that the parameters θd of the domain classifier have been trained to discriminate between the two feature distributions in an optimal way.
This leads us to our idea. At training time, in order to obtain domain-invariant features, we seek the parameters θf of the feature mapping that maximize the loss of the domain classifier (by making the two feature distributions as similar as possible), while simultaneously seeking the parameters θd of the domain classifier that minimize the loss of the domain classifier. In addition, we seek to minimize the loss of the label predictor.

**同时，我们希望使功能f域不变。也就是说，我们想要使分布S（f）= {Gf（x;θf）| x〜S（x）}和T（f）= {Gf（x;θf）| x〜T（x）}是相似的。在协变量偏移假设下，这将使目标域上的标签预测准确性与源域上的标签预测准确性相同[18]。然而，测量分布S（f）和T（f）的不相似性是非平凡的，因为f是高维的，并且随着学习的进展，分布本身也在不断变化。估计不相似性的一种方式是查看域分类器Gd的损失，条件是已经训练了域分类器的参数θd以最佳方式区分两个特征分布。**
**这引出了我们的想法。在训练时，为了获得域不变特征，我们寻找特征映射的参数θf，使域分类器的损失最大化（通过使两个特征分布尽可能相似），同时寻找参数θd最小化域分类器丢失的域分类器。此外，我们寻求最小化标签预测器的损失。**

More formally, we consider the functional:

更正式地说，我们考虑功能：

![](http://118.25.176.50/md_img/论文_深度迁移适应2.JPG)

Here, Ly (·, ·) is the loss for label prediction (e.g. multino- mial), Ld (·, ·) is the loss for the domain classification (e.g. 

logistic), while Liy and Lid denote the corresponding loss functions evaluated at the ith training example. 

Based on our idea, we are seeking the parameters θˆf , θˆy , θˆd that deliver a saddle point of the functional (1): 

这里，Ly（·，·）是标签预测的损失（例如，多项式），Ld（·，·）是域分类的损失（例如，
logy），而Liy和Lid表示在第i个训练样例中评估的相应损失函数。
基于我们的想法，我们正在寻找提供功能（1）的鞍点的参数θf，θy，θd：

![](http://118.25.176.50/md_img/论文_深度迁移适应3.JPG)

At the saddle point, the parameters θd of the domain clas- sifier θd minimize the domain classification loss (since it en- ters into (1) with the minus sign) while the parameters θy of the label predictor minimize the label prediction loss. The feature mapping parameters θf minimize the label predic- tion loss (i.e. the features are discriminative), while maxi- mizing the domain classification loss (i.e. the features are domain-invariant). The parameter λ controls the trade-off between the two objectives that shape the features during learning. 

Below, we demonstrate that standard stochastic gradient solvers (SGD) can be adapted for the search of the saddle point (2)-(3). 

在鞍点处，域分类器θd的参数θd最小化域分类损失（因为它进入（1）带负号），而标签预测器的参数θy使标签预测损失最小化。 特征映射参数θf最小化标签预测损失（即，特征是有区别的），同时最大化域分类损失（即，特征是域不变的）。 参数λ控制在学习期间塑造特征的两个目标之间的权衡。
下面，我们证明标准随机梯度求解器（SGD）可以适用于搜索鞍点（2） - （3）。

### Optimization with backpropagation使用反向传播进行优化

A saddle point (2)-(3) can be found as a stationary point of the following stochastic updates:

**鞍点（2） - （3）可以作为以下随机更新的固定点：**

![](http://118.25.176.50/md_img/论文_深度迁移适应4.JPG)

where μ is the learning rate (which can vary over time). 

The updates (4)-(6) are very similar to stochastic gradi- ent descent (SGD) updates for a feed-forward deep model that comprises feature extractor fed into the label predictor and into the domain classifier. The difference is the    fac- tor in (4) (the difference is important, as without such fac- tor, stochastic gradient descent would try to make features dissimilar across domains in order to minimize the domain classification loss). Although direct implementation of (4)- (6) as SGD is not possible, it is highly desirable to reduce the updates (4)-(6) to some form of SGD, since SGD (and its variants) is the main learning algorithm implemented in most packages for deep learning.

**其中μ是学习率（可以随时间变化）。** 

**更新（4） - （6）非常类似于前馈深度模型的随机梯度下降（SGD）更新，其包括馈送到标签预测器和域分类器中的特征提取器。 不同之处在于（4）中的因素（差异很重要，因为没有这样的因素，随机梯度下降会试图使各个域之间的特征不同，以便最小化域分类损失）。 虽然不能直接实现（4） - （6）作为SGD，但非常希望将更新（4） - （6）减少到某种形式的SGD，因为SGD（及其变体）是主要的学习算法 实施于大多数深度学习套餐。**

Fortunately, such reduction can be accomplished by introducing a special gradient reversal layer (GRL) defined as follows. The gradient reversal layer has no parameters associated with it (apart from the metparameter  , which is not updated by backpropagation). During the forward propagation, GRL acts as an identity transform. During the backpropagation though, GRL takes the gradient from the subsequent level, multiplies it by    and passes it to the preceding layer. Implementing such layer using existing object-oriented packages for deep learning is thus as simple as defining a new layer can be, as defining procedures for forwardprop (identity transform), backprop (multiplying by a constant), and parameter update (nothing) is trivial.

**幸运的是，这种减少可以通过引入如下定义的特殊梯度反转层（GRL）来实现。 梯度反转层没有与之关联的参数（除了参数之外，它不会通过反向传播更新）。 在前向传播期间，GRL充当身份变换。 然而，在反向传播期间，GRL从后续级别获取梯度，将其乘以并将其传递到前一层。 因此，使用现有的面向对象的包进行深度学习来实现这样的层就像定义新层一样简单，因为定义forwardprop（身份变换），backprop（乘以常量）和参数更新（无）的过程是微不足道的。。**

The GRL as defined above is inserted between the feature extractor and the domain classifier, resulting in the architecture depicted in Figure 1. As the backpropagation process passes through the GRL, the partial derivatives of
the loss that is downstream the GRL (i.e. Ld) w.r.t. the layer parameters that are upstream the GRL (i.e. ✓f ) get multiplied by   , i.e. @ Ld is effectively replaced with    @ Ld . @✓f @✓f Therefore, running SGD in the resulting model implements the updates (4)-(6) and converges to a saddle point of (1). Mathematically, we can formally treat the gradient re- versal layer as a “pseudo function” R (x) defined by two (incompatible) equations describing its forward- and back- propagation behaviour:

**如上定义的GRL插在特征提取器和域分类器之间，产生图1所示的体系结构。当反向传播过程通过GRL时，
GRL下游的损失（即Ld）w.r.t。 GRL上游的层参数（即✓f）乘以，即@Ld实际上被@Ld替换。 @✓f@✓f因此，在结果模型中运行SGD实现更新（4） - （6）并收敛到（1）的鞍点。 在数学上，我们可以正式将梯度反射层视为由两个（不兼容的）方程定义的“伪函数”R（x），描述其前向和后向传播行为：**

![](http://118.25.176.50/md_img/论文_深度迁移适应5.JPG)

where I is an identity matrix. We can then define the objective “pseudo-function” of (✓f , ✓y , ✓d ) that is being optimized by the stochastic gradient descent within our method:

**我是一个单位矩阵。 然后我们可以定义（✓f，✓y，✓d）的目标“伪函数”，它通过我们方法中的随机梯度下降进行优化：**

![](http://118.25.176.50/md_img/论文_深度迁移适应6.JPG)

Running SGD for (9) thus leads to the emergence of fea- tures that are domain-invariant and discriminative at the same time. After the learning, the label predictor y(x) = Gy (Gf (x; ✓f ); ✓y ) can be used to predict labels for samples from the target domain (as well as from the source domain).
The simple learning procedure outlined above can be re- derived/generalized along the lines suggested in [9] (see Appendix).

**因此，为（9）运行SGD会导致同时具有域不变性和区别性的特征的出现。 在学习之后，标签预测器y（x）= Gy（Gf（x;✓f）;✓y）可用于预测来自目标域（以及源域）的样本的标签。
上面概述的简单学习过程可以按照[9]中建议的方式重新推导/推广（见附录）。**

## Experiments实验

### Datasets数据集

In this report, we validate our approach within a set experiments on digit image classification. In each case, we train on the source dataset and test on a different target domain dataset, with considerable shifts between domains (see Fig- ure 2). Overall, we have six datasets involved into the experiments that are described below.

**在本报告中，我们在数字图像分类的集合实验中验证了我们的方法。 在每种情况下，我们在源数据集上进行训练并在不同的目标域数据集上进行测试，域之间有相当大的变化（见图2）。 总的来说，我们有六个数据集参与下面描述的实验。**

![](http://118.25.176.50/md_img/论文_深度迁移适应7.JPG)

Table 1. Classification accuracies for digit image classifications for different source and target domains. In the case of MNIST, four different versions of the dataset were used: the original one, its dilated version (D), max-blended numbers over background (max, BG) and difference-blended numbers over background (| |, BG). The first row corresponds to the lower performance bound (i.e. if no adaptation is performed). The last row corresponds to training on the target domain data with known class labels (upper bound on the DA performance). For each of the two DA methods (ours and [5]) we show how much of the gap between the lower and the upper bounds was covered (in brackets). For all five cases, our approach outperforms [5] considerably, and covers a big portion of the gap.

**表1.不同源和目标域的数字图像分类的分类准确度。 在MNIST的情况下，使用了四个不同版本的数据集：原始数据集，其扩张版本（D），背景上的最大混合数字（最大值，BG）和背景上的差异混合数字（| |，BG）。 第一行对应于较低的性能界限（即，如果不执行自适应）。 最后一行对应于具有已知类标签的目标域数据的训练（DA性能的上限）。 对于两种DA方法（我们的和[5]）中的每一种，我们都显示了下限和上限之间的差距（括号内）。 对于所有五种情况，我们的方法在很大程度上优于[5]，并且覆盖了很大一部分差距。**

The first four datasets are the well-known MNIST dataset [12] and its modifications. The modifications in- clude:

- MNIST (D): Binary dilation with a 3⇥3 all-ones struc- turing element. This operation makes strokes thicker and may fill small holes thereby introducing additional challenge to the classification task.
- MNIST (max, BG): Blending white digits over the patches randomly extracted from photos (BSDS500). While leaving meaningful pixels as they are, this mod- ification adds strong background clutter.
- MNIST (| |, BG): Difference-blending digits over the patches randomly extracted from photos (BSDS500). This operation is formally defined for two images where i,j are the coordinates of a pixel and k is a channel index. In other words, an output sample is produced by taking a patch from a photo and inverting its pixels at positions cor- responding to the pixels of a digit. For a human the classification task becomes only slightly harder com- pared to the original dataset (the digits are still clearly distinguishable) whereas for a CNN trained on MNIST this domain is quite distinct: the background and the strokes are no longer constant.

**前四个数据集是众所周知的MNIST数据集[12]及其修改。修改包括：**

- **MNIST（D）：具有3⇥3全1结构元素的二进制扩张。该操作使笔划更粗并且可填充小孔，从而对分类任务引入额外的挑战。**
- **MNIST（max，BG）：在从照片（BSDS500）中随机提取的色块上混合白色数字。虽然保留了有意义的像素，但这种修改增加了强烈的背景混乱。**
- **MNIST（| |，BG）：从照片（BSDS500）中随机提取的补丁上的差异混合数字。该操作被正式定义为两个图像，其中i，j是像素的坐标，k是通道索引。换句话说，通过从照片中取出贴片并在与数字像素相对应的位置处反转其像素来产生输出样本。对于人类而言，分类任务与原始数据集相比变得稍微有点难度（数字仍然可以清楚地区分），而对于在MNIST上训练的CNN，这个域非常不同：背景和笔画不再是恒定的。**

 The last two datasets is the well-known Street View House Number (SVHN) dataset [14] and a new synthetic dataset Syn. Numbers of 500,000 images generated by ourselves from Windows fonts by varying the text (that includes different one-, two-, and three-digit numbers), positioning, orientation, background and stroke colors, and the amount of blur. The degrees of variation were chosen manually to simulate SVHN, however the two datasets are still rather distinct, the biggest difference being the structured clutter in the background of SVHN images.

**最后两个数据集是众所周知的街景房号（SVHN）数据集[14]和新的合成数据集Syn。 通过改变文本（包括不同的一位，两位和三位数字），定位，方向，背景和笔触颜色以及模糊量，我们自己从Windows字体生成的500,000张图像的数量。 手动选择变化程度来模拟SVHN，但是两个数据集仍然相当明显，最大的区别在于SVHN图像背景中的结构化混乱。**

### beasline基线

In each experiment, we compare the results of our approach with the two natural baselines, i.e. training on the source do- main without adaptation (a lower bound on any reasonable DA method) and training on the target domain while exploiting target domain labels (an upper bound on DA meth- ods, assuming that target data are abundant and the shift between the domains is considerable).
In addition, we compare our approach against the re- cently proposed unsupervised DA method based on sub- space alignment (SA) [5]. We detail the protocol for the use of SA further below.
The SA algorithm has one important free parameter, namely the number of principal components. In each of the experiments, we give this baseline an advantage by picking this value from the range {2, . . . , 60}, so that the performance on the target domain is maximized.

**在每个实验中，我们将我们的方法的结果与两个自然基线进行比较，即在没有适应的情况下对源头进行训练（任何合理的DA方法的下限）以及在利用目标域标签时对目标域进行训练（ DA方法的上限，假设目标数据丰富且域之间的转移相当大）。
此外，我们将我们的方法与最近提出的基于子空间对齐（SA）的无监督DA方法进行了比较[5]。 我们在下面详细介绍了使用SA的协议。
SA算法有一个重要的自由参数，即主成分的数量。 在每个实验中，我们通过从范围{2，...中选择此值来为此基线提供优势。。。 ，60}，以便最大化目标域上的性能。**

### CNN architectures CNN架构

Two different architectures were used in our experiments (Figure 3). We employ the smaller one if the source domain is MNIST and the bigger one otherwise. The chosen architectures are fairly standard for these datasets in terms of feature extractor and label predictor parts: the “MNIST” design is inspired by the classical LeNet-5 [12], while the second CNN is adopted from [19]. The domain classifier branch in both cases is somewhat arbitrary – the effect of changing its design is yet to be analyzed.
As for the loss functions, we set Ly and Ld to be logistic regression cost and binomial cross-entropy respectively.

**在我们的实验中使用了两种不同的架构（图3）。 如果源域是MNIST，我们使用较小的域，否则使用较大的域。 根据特征提取器和标签预测器部分，所选择的体系结构对于这些数据集是相当标准的：“MNIST”设计的灵感来自经典的LeNet-5 [12]，而第二个CNN则采用[19]。 两种情况下的域分类器分支都有些武断 - 改变其设计的效果尚待分析。
对于损失函数，我们将Ly和Ld分别设置为逻辑回归成本和二项式交叉熵。**

### Training procedure. 训练程序

The model is trained on 128-element batches of 32 ⇥ 32 color patches (we replicate channels for the original MNIST). No preprocessing is done except for the overall mean subtraction. A half of each batch is populated by the samples from the source domain (with known labels), the rest is comprised of the target domain (with unknown labels). 

We use stochastic gradient descent with 0.9 momentum and the learning rate annealing described by the following formula: 

**该模型采用128个元素批次的32×32色块进行训练（我们复制原始MNIST的通道）。 除了整体平均减法之外，不进行预处理。 每个批次的一半由来自源域的样本（具有已知标签）填充，其余部分由目标域（具有未知标签）组成。**

**我们使用具有0.9动量的随机梯度下降和由以下公式描述的学习速率退火: **

![](http://118.25.176.50/md_img/论文_深度迁移适应8.JPG)

where p is the training progress linearly changing from 0 to 1, μ0 = 0.01, ↵ = 10 and   = 0.75 (the schedule was opti- mized to promote convergence and low error on the source domain).
In order to suppress noisy signal from the domain clas- sifier at the early stages of the training procedure instead of fixing the adaptation factor  , we gradually change it from 0 to 1 using the following schedule:

**其中p是从0到1线性变化的训练进度，μ0= 0.01，↵= 10和= 0.75（优化时间表以促进源域上的收敛和低误差）。
为了在训练过程的早期阶段抑制来自域分类器的噪声信号而不是固定适应因子，我们使用以下时间表逐渐将其从0更改为1：**

![](http://118.25.176.50/md_img/论文_深度迁移适应9.JPG)

where   was set to 10 in all experiments (the schedule was not optimized/tweaked).
Following [19] we also use dropout and `2-norm restriction when we train the SVHN architecture.
For the SA baseline, we consider the activations of the last hidden layer in the label predictor (before the final linear classifier) as descriptors/features, and learn the mapping between the source and the target domains [5].
Since the SA baseline requires to train a new classifier after adapting the features, and in order to put all the compared methods on an equal footing, we retrain the last layer of the label predictor using a standard linear SVM [4] for all four compared methods (including ours; the performance on the target domain remains approximately the same after the retraining).

**在所有实验中，其中设置为10（计划未优化/调整）。
在[19]之后，当我们训练SVHN架构时，我们也使用了dropout和`2-norm限制。
对于SA基线，我们将标签预测器中最后一个隐藏层的激活（在最终线性分类器之前）视为描述符/特征，并学习源域和目标域之间的映射[5]。
由于SA基线需要在调整特征后训练新的分类器，并且为了使所有比较的方法处于相同的基础，我们使用标准线性SVM重新训练标签预测器的最后一层[4]用于所有四个比较 方法（包括我们的;在重新训练后，目标领域的表现保持不变）。**

### Visualizations可视化

We use t-SNE [22] projection to visualize feature distributions at different points of the network, while color-coding the domains (Figure 4). Overall, we observe quite strong correlation between the success of the adaptation in terms of the classification accuracy for the target domain, and the amount of discrepancy between the domain distributions in our visualizations.

**我们使用t-SNE [22]投影来可视化网络不同点的特征分布，同时对域进行颜色编码（图4）。 总体而言，我们观察到在目标域的分类准确性方面的适应成功与我们可视化中的域分布之间的差异量之间存在非常强的相关性。**

### Results 结果

We test our approach as well as the baselines for six different domain pairs. The results obtained by the composition of the feature extractor and the label predictor for the three baseline methods and our approach are summarized in Table 1 and are discussed below.

**我们测试了我们的方法以及六个不同域对的基线。 通过特征提取器的组成和三种基线方法的标签预测器以及我们的方法获得的结果总结在表1中并在下面讨论。**

#### MNIST ->its variations

In the first three experiments, we deal with the MNIST dataset: a classifier is trained on the original dataset while being adapted to perform well on a particular modification of the source domain. The three target domains can be ordered in terms of the similarity to the source domain (which can be judged based on the performance of the classifier trained on the source domain and applied to the target domain). As expected, domain adaptation is easiest for the target domain that is closest to the source (MNIST (D)), and our method is able to cover three quarters of the performance gap between the source-trained and target-trained classifiers (i.e. lower and upper bounds).

**在前三个实验中，我们处理MNIST数据集：在原始数据集上训练分类器，同时适应在源域的特定修改上表现良好。 可以根据与源域的相似性来对三个目标域进行排序（可以基于在源域上训练并应用于目标域的分类器的性能来判断）。 正如所料，对于最接近源（MNIST（D））的目标域，域自适应最容易，并且我们的方法能够覆盖源训练分类器和目标训练分类器之间的性能差距的四分之三（即较低） 和上限）。**

The adaptation task is harder in the case of digits blended over the color background. Although samples from these domains have significant background clutter, our approach succeeded at intermixing the features (Figure 4), which led to very successful adaptation results (considering that the adaptation is unsupervised). The performance of the unsupervised DA method [5] for all three datasets is much more modest, thus highlighting the difficulty of the adaptation task.

**在彩色背景上混合数字的情况下，适应任务更难。 尽管来自这些域的样本具有显着的背景杂波，但我们的方法成功地混合了特征（图4），这导致非常成功的适应结果（考虑到适应性是无监督的）。 所有三个数据集的无监督DA方法[5]的性能要更加适度，从而突出了适应任务的难度。**

![](http://118.25.176.50/md_img/论文_深度迁移适应10.JPG)

Figure 3. CNN architectures used in the experiments. Boxes correspond to transformations applied to the data. Color-coding is the same as in Figure 1. See Section 4 for details. 

**图3.实验中使用的CNN架构。 框对应于应用于数据的转换。 颜色编码与图1中的相同。有关详细信息，请参阅第4节。**

#### Synthetic numbers合成数字 -> SVHN

To address a common scenario of training on synthetic images and testing on challenging real images, we use SVHN as a target domain and synthetic digits as a source. The proposed backpropagation-based technique works well covering two thirds of the gap between training with source data only and training on target domain data with known target labels. In contrast, [5] does not result in any significant improvement in the classification accuracy, thus highlighting that the adaptation task is even more challenging than in the case of MNIST experiments.

**为了解决合成图像训练和挑战性真实图像测试的常见场景，我们使用SVHN作为目标域，使用合成数字作为源。 所提出的基于反向传播的技术很好地覆盖了仅有源数据的训练与具有已知目标标签的目标域数据的训练之间的差距的三分之二。 相比之下，[5]并未导致分类准确性的任何显着改善，因此强调适应任务比MNIST实验的情况更具挑战性。**

#### MNIST <—>SVHN

Finally, we test our approach on the two most distinct domains, namely, MNIST and SVHN. Training on SVHN even without adaptation is challenging — classification error stays high during the first 150 epochs. In order to avoid ending up in a poor local minimum we, therefore, do not use learning rate annealing here. Obviously, the two directions (MNIST-to-SVHN and SVHN-to-MNIST) are not equally difficult. As SVHN is more diverse, a model trained on SVHN is expected to be more generic and to perform reasonably on the MNIST dataset. This, indeed, turns out to be the case and is supported by the appearance of the feature distributions. We observe a quite strong separation between the domains when we feed them into the CNN trained solely on MNIST, whereas for the SVHN-trained network the features are much more intermixed. This difference probably explains why our method succeeded in improving the performance by adaptation in the SVHN ! MNIST scenario (see Table 1) but not in the opposite direction (SA is not able to perform adaptation in this case either). Unsupervised adaptation from MNIST to SVHN thus remains a challenge to be addressed in the future work.

**最后，我们在两个最不同的域上测试我们的方法，即MNIST和SVHN。即使没有适应性，对SVHN的培训也具有挑战性 - 在前150个时期，分类错误仍然很高。因此，为了避免在较差的局部最小值，我们不要在这里使用学习率退火。显然，这两个方向（MNIST-to-SVHN和SVHN-to-MNIST）并不是同样困难的。由于SVHN更加多样化，因此预计在SVHN上训练的模型将更加通用并且在MNIST数据集上合理地执行。事实上，事实证明是这种情况，并且由特征分布的外观支持。当我们将它们馈送到仅在MNIST上训练的CNN时，我们观察到域之间非常强烈的分离，而对于SVHN训练的网络，这些特征更加混杂。这种差异可能解释了为什么我们的方法通过SVHN中的适应性成功地提高了性能！ MNIST场景（见表1）但不是相反的方向（在这种情况下SA也不能进行调整）。因此，从MNIST到SVHN的无监督调整仍然是未来工作中需要解决的挑战。**

## Discussion 讨论

We have proposed a new approach to unsupervised domain adaptation of deep feed-forward architectures, which allows large-scale training based on large amount of annotated data in the source domain and large amount of unannotated data in the target domain. Similarly to many previous shallow and deep DA techniques, the adaptation is achieved through aligning the distributions of features across the two domains. However, unlike previous approaches, the alignment is accomplished through standard backpropagation training. The approach is therefore rather scalable, and can be implemented using any deep learning package.
In the experiments with digit image classification, the approach demonstrated its efficiency, significantly outperforming a state-of-the-art unsupervised DA method. Further evaluation on larger-scale tasks constitutes the immediate future work. It is also interesting whether the approach can benefit from a good initialization of the feature extractor. For this, a natural choice would be to use deep autoencoder/deconvolution network trained on both domains (or on the target domain) in a similar vein to [6, 17], effectively using [6, 17] as an initialization to our method.

**我们已经提出了一种新的深度前馈体系结构的无监督域自适应方法，它允许基于源域中的大量注释数据和目标域中的大量未注释数据进行大规模训练。与许多先前的浅层和深层DA技术类似，通过在两个域中对齐特征的分布来实现自适应。然而，与先前的方法不同，通过标准反向传播训练完成对准。因此，该方法具有相当的可扩展性，并且可以使用任何深度学习包来实现。
在数字图像分类的实验中，该方法证明了其效率，明显优于最先进的无监督DA方法。对大规模任务的进一步评估构成了近期的工作。这种方法是否可以从特征提取器的良好初始化中受益，这也很有趣。为此，一个自然的选择是使用在两个域（或目标域）上训练的深度自动编码器/反卷积网络，与[6,17]类似，有效地使用[6,17]作为我们方法的初始化。**

![](http://118.25.176.50/md_img/论文_深度迁移适应11.JPG)

![](http://118.25.176.50/md_img/论文_深度迁移适应12.JPG)

![](http://118.25.176.50/md_img/论文_深度迁移适应13.JPG)

Figure 4. The effect of adaptation on the distribution of the extracted features. The figure shows t-SNE [22] visualizations of the CNN’s activations (a) in case when no adaptation was performed and (b) in case when our adaptation procedure was incorporated into training. Blue points correspond to the source domain examples, while red ones correspond to the target domain. In all cases, the adaptation in our method makes the two distributions of features much closer.

**图4.适应对提取特征分布的影响。 该图显示了CNN激活的t-SNE [22]可视化（a）在没有进行适应的情况下和（b）在我们的适应程序被纳入训练的情况下。 蓝点对应于源域示例，而红色对应于目标域。 在所有情况下，我们的方法中的自适应使得两个特征的分布更加接近。**