---
title: Domain Adaptation via Transfer Component Analysis论文翻译
date: 2018-11-28 22:58:54
categories: 论文翻译
tags: [论文,机器学习,深度学习,迁移学习]
---

<Excerpt in index | 首页摘要> 

本文提出一种类似于PCA的降维方法，TCA

<!-- more -->

<The rest of contents | 余下全文>

# Domain Adaptation via Transfer Component Analysis

## 摘要

Domain adaptation solves a learning problem in a target domain by utilizing the training data in a different but related source domain. Intuitively, discovering a good feature representation across domains is crucial. In this paper, we propose to ﬁnd such a representation through a new learning method, transfer component analysis (TCA), for domain adaptation. TCA tries to learn some transfer components across domains in a Reproducing Kernel Hilbert Space (RKHS) using Maximum Mean Discrepancy (MMD). In the subspace spanned by these transfer components, data distributions in different domains are close to each other. As a result, with the new representations in this subspace, we can apply standard machine learning methods to train classiﬁers or regression models in the source domain for use in the target domain. The main contribution of our work is that we propose a novel feature representation in which to perform domain adaptation via a new parametric kernel using feature extraction methods, which can dramatically minimize the distance between domain distributions by projecting data onto the learned transfer components. Furthermore, our approach can handle largedatsets and naturally lead to out-of-sample generalization. The effectiveness and efﬁciency of our approach in are veriﬁed by experiments on two real-world applications: cross-domain indoor WiFi localization and cross-domain text classiﬁcation.

**域自适应通过利用不同但相关的源域中的训练数据来解决目标域中的学习问题。直觉上，发现跨域的良好特征表示是至关重要的。在本文中，我们建议通过一种新的学习方法，转移成分分析（TCA），为域适应找到这样的表示。 TCA尝试使用最大平均差异（MMD）在再生核希尔伯特空间（RKHS）中跨域学习一些转移组件。在由这些传输组件跨越的子空间中，不同域中的数据分布彼此接近。因此，通过该子空间中的新表示，我们可以应用标准机器学习方法来训练源域中的分类器或回归模型，以用于目标域。我们工作的主要贡献在于我们提出了一种新颖的特征表示，其中使用特征提取方法通过新的参数核执行域适应，这可以通过将数据投影到学习的传输组件上来极大地最小化域分布之间的距离。此外，我们的方法可以处理大数据集，并自然导致样本外泛化。我们的方法的有效性和效率通过两个实际应用的实验来验证：跨域室内WiFi定位和跨域文本分类。**

## 介绍

Domain adaptation aims at adapting a classiﬁer or regression model trained in a source domain for use in a target domain, where the source and target domains may be different but related. This is particularly crucial when labeled data are in short supply in the target domain. For example, in indoor WiFi localization, it is very expensive to calibrate a localization model in a large-scale environment. However, the WiFi signal strength may be a function of time, device or space, depending on dynamic factors. To reduce the re-calibration effort, we might want to adapt a localization model trained in one time period (the source domain) for a new time period (the target domain), or to adapt the localization model trained on one mobile device (the source domain) for a new mobile device (the target domain). However, the distributions of WiFi data collected over time or across devices may be very different, hence domain adaptation is needed [Yang et al., 2008]. Another example is sentiment classiﬁcation. To reduce the effort of annotating reviews for various products, we might want to adapt a learning system trained on some types of products (the source domain) for a new type of product (the target domain). However, terms used in the reviews of different types of products may be very different. As a result, distributions of the data over different types of products may be different and thus domain adaptation is again needed [Blitzer et al., 2007]. 

**域适应旨在调整在源域中训练的分类器或回归模型以用于目标域，其中源域和目标域可以是不同的但是相关的。当标记数据在目标域中供不应求时，这尤其重要。例如，在室内WiFi定位中，在大规模环境中校准定位模型是非常昂贵的。然而，取决于动态因素，WiFi信号强度可以是时间，设备或空间的函数。为了减少重新校准工作量，我们可能希望调整在一个时间段（源域）中训练的本地化模型用于新的时间段（目标域），或者调整在一个移动设备上训练的本地化模型（源域）用于新的移动设备（目标域）。然而，随时间或跨设备收集的WiFi数据的分布可能非常不同，因此需要域适应[Yang等人，2008]。另一个例子是情绪分类。为了减少为各种产品注释评论的工作量，我们可能希望调整针对某种类型的产品（源域）培训的学习系统，以用于新类型的产品（目标域）。但是，在审查不同类型的产品时使用的术语可能非常不同。结果，不同类型产品上的数据分布可能不同，因此再次需要域适应[Blitzer等，2007]。**

A major computational problem in domain adaptation is how to reduce the difference between the distributions of source and target domain data. Intuitively, discovering a good feature representation across domains is crucial. A good feature representation should be able to reduce the difference in distributions between domains as much as possible, while at the same time preserving important (geometric or statistical) properties of the original data.

Recently, several approaches have been proposed to learn a common feature representation for domain adaptation [Daum´e III, 2007; Blitzer et al., 2006]. Daum´ e III [2007] proposed a simple heuristic nonlinear mapping function to map the data from both source and target domains to a highdimensional feature space, where standard machine learning methods are used to train classiﬁers. Blitzer et al.[2006] proposed the so-called structural correspondencelearning (SCL) algorithmtoinducecorrespondencesamongfeaturesfromthe different domains. This method depends on the heuristic selections of pivot features that appear frequently in both domains. Although it is experimentally shown that SCL can reduce the differencebetween domainsbased on theA-distance measure [Ben-David et al., 2007], the heuristic criterion of pivot feature selection may be sensitive to different applications. Pan et al.[2008] proposed a new dimensionality reduction method, Maximum Mean Discrepancy Embedding (MMDE), for domain adaptation. The motivation of MMDE is similar to our proposed work. It also aims at learning a shared latent space underlying the domains where distance between distributions can be reduced. However, MMDE suffers from two major limitations: (1) MMDE is transductive, and doesnot generalizeto out-of-samplepatterns; (2) MMDE learns the latent space by solving a semi-deﬁnite program (SDP), which is a very expensive optimization problem.

**域自适应中的主要计算问题是如何减少源域数据和目标域数据的分布之间的差异。直觉上，发现跨域的良好特征表示是至关重要的。良好的特征表示应该能够尽可能地减少域之间的分布差异，同时保留原始数据的重要（几何或统计）属性。**

**最近，已经提出了几种方法来学习域适应的共同特征表示[Daum'e III，2007; Blitzer等，2006]。 Daum'e III [2007]提出了一种简单的启发式非线性映射函数，用于将来自源域和目标域的数据映射到高维特征空间，其中标准机器学习方法用于训练分类器。 Blitzer等人[2006]提出了所谓的结构对应学习（SCL）算法来引导来自不同领域的对应特征。此方法取决于在两个域中频繁出现的透视功能的启发式选择。虽然通过实验证明SCL可以减少基于A距离测量的域之间的差异[Ben-David等，2007]，但是枢轴特征选择的启发式标准可能对不同的应用敏感。潘等人[2008]提出了一种新的降维方法，最大均值差异嵌入（MMDE），用于域自适应。 MMDE的动机与我们提出的工作类似。它还旨在学习可以减少分布之间距离的域下的共享潜在空间。然而，MMDE有两个主要的局限性：（1）MMDE是转导性的，并不是一般性的样本外模式; （2）MMDE通过求解半定义程序（SDP）来学习潜在空间，这是一个非常昂贵的优化问题。**

In this paper, we propose a new feature extraction approach, called transfer component analysis (TCA), for domain adaptation. It tries to learn a set of common transfer components underlying both domains such that the difference in distributions of data in the different domains, when projected onto this subspace, can be dramatically reduced. Then, standard machine learning methods can be used in this subspace to train classiﬁers or regression models across domains. More speciﬁcally, if two domains are related to each other, there may exist several common components (or latent variables) underlying them. Some of these components may cause the data distributions between domains to be different, while others may not. Some of these components may capture the intrinsic structure underlying the original data, while others may not. Our goal is to discover those components that do not cause distribution change across the domains and capture the structure of the original data well. We will show in this paper that, compared to MMDE, TCA is much more efﬁcient and can handlethe out-of-sampleextension problem.

**在本文中，我们提出了一种新的特征提取方法，称为传递分量分析（TCA），用于域自适应。它试图学习两个域下面的一组公共传递组件，这样当投影到这个子空间时，不同域中数据分布的差异可以大大减少。然后，可以在该子空间中使用标准机器学习方法来训练跨域的分类器或回归模型。更具体地说，如果两个域彼此相关，则可能存在几个共同的组件（或潜在变量）。其中一些组件可能导致域之间的数据分布不同，而其他组件可能不同。这些组件中的一些可以捕获原始数据的内在结构，而其他组件可能不捕获。我们的目标是发现那些不会导致跨域分布变化的组件，并很好地捕获原始数据的结构。我们将在本文中表明，与MMDE相比，TCA更有效，并且可以处理样本外扩展问题。**

The rest of the paper is organizedas follows. Section 2 ﬁrst describes the problem statement and preliminaries of domain adaptation. Our proposed method is presented in Section 3. We then review some related works in Section 4. In Section 5, we conduct a series of experiments on indoor WiFi localization and text classiﬁcation. The last section gives some conclusive discussions. In the sequel, A > 0 (resp. A >=0) means that the matrix A is symmetric and positive deﬁnite (pd) (resp. positive semideﬁnite (psd)). Moreover, the transpose of vector / matrix (in both the input and feature spaces) is denoted by the superscript ^T, A† is the pseudo-inverse of the matrix A, and tr(A) denotes the trace of A.

**本文的其余部分安排如下。 第2节首先描述了域适应的问题陈述和预备。 我们提出的方法在第3节中介绍。然后我们回顾第4节中的一些相关工作。在第5节中，我们进行了一系列关于室内WiFi定位和文本分类的实验。 最后一节给出了一些结论性的讨论。 在下文中，A> 0（分别为A> = 0）意味着矩阵A是对称的并且是正定义（pd）（分别是正半无限（psd））。 此外，矢量/矩阵的转置（在输入和特征空间中）由上标T表示，A†是矩阵A的伪逆，tr（A）表示A的轨迹。**

## Preliminaries of Domain Adaptation 领域适应准备工作

In this paper, we focus on the setting where the target domain has no labeled training data, but has plenty of unlabeled data. We also assume that some labeled data DS are available in a source domain, while only unlabeled data DT are available in the target domain. We denote the source domain data as DS = {(xS1,yS1),...,(xSn1 ,ySn1 )}, wherexSi ∈Xis the input and ySi ∈Yis the corresponding output. Similarly, we denote the target domain data as DT = {xT1,...,xTn2 }, where the input xTi is also in X. LetP(XS) and Q(XT) (or P and Q for short) be the marginal distributions of XS and XT, respectively. In general, P and Q can be different. Our task is then to predict the labels yTi’s corresponding to the inputs xTi’s in the target domain. The key assumption in a typical domain adaptation setting is that P != Q, but P(YS|XS)=P(YT|XT) .

**在本文中，我们关注目标域没有标记的训练数据但是有大量未标记数据的设置。 我们还假设某些标记数据DS在源域中可用，而目标域中只有未标记数据DT可用。 我们将源域数据表示为DS = {（xS1，yS1），...，（xSn1，ySn1）}，其中xSi∈X是输入，ySi∈Y是相应的输出。 类似地，我们将目标域数据表示为DT = {xT1，...，xTn2}，其中输入xTi也在X.LetP（XS）和Q（XT）（或简称P和Q）是边际 XS和XT的分布。 通常，P和Q可以不同。 然后我们的任务是预测与目标域中的输入xTi相对应的标签yTi。 典型域适配设置中的关键假设是P！= Q，但P（YS | XS）= P（YT | XT）。**

### Maximum Mean Discrepancy 最大平均差异

Many criteria, such as the Kullback-Leibler (KL) divergence, can be used to estimate the distance between distributions.
However, many of these criteria are parametric, since an intermediate density estimate is usually required. To avoid such a non-trivial task, a non-parametric distance estimate between distributions is more desirable. Recently, Borgwardt et al.[2006]proposedtheMaximum Mean Discrepancy (MMD) as a relevant criterion for comparing distributions based on the Reproducing Kernel Hilbert Space (RKHS). Let X = {x1,...,xn1} and Y = {y1,...,yn2} be random variable sets with distributions P and Q. The empirical estimate of the distance between P and Q, as deﬁned by MMD, is where H is a universal RKHS [Steinwart, 2001], andφ : X→H .

Therefore, the distance between distributions of two samples can be well-estimated by the distance between the means of the two samples mapped into a RKHS.

**许多标准，例如Kullback-Leibler（KL）散度，可用于估计分布之间的距离。 然而，这些标准中的许多是参数的，因为通常需要中间密度估计。 为了避免这种非平凡的任务，更期望分布之间的非参数距离估计。 最近，Borgwardt等人[2006]提出了最大平均差异（MMD）作为比较基于再生核希尔伯特空间（RKHS）的分布的相关标准。 设X = {x1，...，xn1}和Y = {y1，...，yn2}是具有分布P和Q的随机变量集.P和Q之间距离的经验估计，如MMD所定义， H是通用RKHS [Steinwart，2001]，φ：X→H。**

![img](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_1.jpg)

**因此，可以通过映射到RKHS的两个样本的平均值之间的距离来很好地估计两个样本的分布之间的距离。**

## Transfer Component Analysis  转移成分分析

Based on the inputs {xSi} and outputs {ySi} from the source domain, and the inputs {xTi} from the target domain, our task is to predict the unknown outputs {yTi} in the target domain. The general assumption in domain adaptation is that the marginal densities, P(XS) and Q(XT), are very different. In this section, we attempt to ﬁnd a common latent representation for both XS and XT that preserves the data conﬁguration of the two domains after transformation. Let the desired nonlinear transformation be φ : X→H. Let X' S = {x'Si} = {φ(xSi)}, X'T = {x'Ti} = {φ(xTi)} and X' = X' S∪X' T be the transformedinput sets from the source, target and combined domains, respectively. Then, we desire that P'(X' S)=Q'(X' T).

**基于来自源域的输入{xSi}和输出{ySi}以及来自目标域的输入{xTi}，我们的任务是预测目标域中的未知输出{yTi}。 域适应的一般假设是边际密度P（XS）和Q（XT）非常不同。 在本节中，我们尝试为XS和XT找到一个共同的潜在表示，它保留了转换后两个域的数据配置。 设期望的非线性变换为φ：X→H。 令X'S = {x'Si} = {φ（xSi）}，X'T = {x'Ti} = {φ（xTi）}和X'=X'S∪X'T是来自的变换输入集 分别是源，目标和组合域。 然后，我们希望P'（X'S）= Q'（X'T）。	**

Assuming that φ is the feature map induced by a universal kernel. As shown in Section 2.1, the distance between two distributions P and Q can be empirically measured by the (squared) distance between the empirical means of the two domains:

**假设φ是通用内核引起的特征映射。 如2.1节所示，两个分布P和Q之间的距离可以通过两个域的经验方法之间的（平方）距离来经验地测量：**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_2.jpg)

Therefore, a desired nonlinear mapping φ can be found by minimizing this quantity. However, φ is usually highly nonlinear and a direct optimization of (2) can get stuck in poor local minima. We thus need to ﬁnd a new approach, based on the following assumption. 

**因此，可以通过最小化该量来找到期望的非线性映射。 然而，φ通常是高度非线性的，并且（2）的直接优化可能陷入较差的局部最小值。 因此，我们需要根据以下假设找到一种新方法。**

The key assumption in the proposed domain adaptation setting is that P = Q, butP(YS|φ(XS)) = P(YT|φ(XT)) under a transformation mapping φ on the input. In Section 3.1, we ﬁrst revisit Maximum Mean DiscrepancyEmbedding(MMDE)whichproposedtolearnthekernel matrix K corresponding to the nonlinear mapping φ by solving a SDP optimization problem. In Section 3.2, we then propose a factorization of the kernel matrix for MMDE. An efﬁcient eigendecomposition algorithm for kernel learning and computational issues are discussed in Sections 3.3 and 3.4.

**所提出的域自适应设置中的关键假设是在输入上的变换映射φ下P <= Q，但是P（YS |φ（XS））= P（YT |φ（XT））。 在3.1节中，我们首先重新讨论了最大平均差异嵌入（MMDE），它通过解决SDP优化问题提出了对应于非线性映射φ的核心矩阵K. 在3.2节中，我们提出了MMDE核矩阵的分解。 第3.3节和第3.4节讨论了用于内核学习和计算问题的高效特征分解算法。**

### Kernel Learning for Domain Adaptation 域适应的内核学习

Instead of ﬁnding the nonlinear transformation φ explicitly, Pan et al.[2008] proposed to transform this problem as a kernel learning problem. By virtue of the kernel trick, (i.e., k(xi,x j)=φ(xi)'φ(xj)), thedistance between theempirical means of the two domains in (2) can be written as: 

**Pan et al。[2008]没有明确地发现非线性变换。 建议将此问题转化为内核学习问题。 借助于核技巧（即k（xi，x j）=φ（xi）'φ（xj）），（2）中两个域的经验平均值之间的距离可写为：**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_3.jpg)

is a (n1 + n2) × (n1 + n2) kernel matrix, KS,S, KT,T and KS,T respectively are the kernel matrices deﬁned by k on the data in the source domain, target domain, and cross domains; and L =[Lij] > 0 with Lij =1/n1^2 if xi,x j ∈ XS; Lij = 1 n2 ^2 if  xi,x j ∈ XT; otherwise,− 1/ n1n2 . 

**是（n1 + n2）×（n1 + n2）个核矩阵，KS，S，KT，T和KS，T分别是由k对源域，目标域和交叉域中的数据定义的核矩阵; 如果xi，xj∈XS，L = [Lij]> 0，Lij = 1 / n1 ^ 2; 如果xi，xj∈XT，Lij = 1 n2 ^ 2; 否则， - 1 / n1n2。**

In the transductive setting, learning the kernel k(·,·) can be solved by learning the kernel matrix K instead. In [Pan et al., 2008], the resultant kernel matrix learning problem is formulated as a semi-deﬁnite program (SDP). Principal Component Analysis (PCA) is then applied on the learned kernel matrix to ﬁnd a low-dimensionallatent space across domains. This is referred to as Maximum Mean Discrepancy Embedding (MMDE).

**在转换设置中，学习核k（·，·）可以通过学习核矩阵K来解决。 在[Pan et al。，2008]中，得到的核矩阵学习问题被公式化为半有限程序（SDP）。 然后将主成分分析（PCA）应用于学习的核矩阵以找到跨域的低维空间。 这被称为最大平均差异嵌入（MMDE）。**

### Parametric Kernel Map for Unseen Patterns 看不见的模式的参数核心图

There are several limitations of MMDE. First, it is transductive and cannot generalize on unseen patterns. Second, the criterion (3) requires K to be positive semi-deﬁnite and the resultant kernel learning problem has to be solved by expensive SDP solvers. Finally, in order to construct lowdimensional representations of X S and X T, the obtained K has to be further post-processed by PCA. This may potentially discard useful information in K. In this paper, we propose an efﬁcient method to ﬁnd a nonlinear mapping φ based on kernel feature extraction. It avoids the use of SDP and thus its high computationalburden. Moreover, the learned kernel k can be generalized to out-of-sample patterns directly. Besides, instead of using a two-step approach as in MMDE, we propose a uniﬁed kernel learning method which utilizes an explicit low-rank representation. 

**MMDE有几个局限性。 首先，它是转换性的，不能概括于看不见的模式。 其次，标准（3）要求K为正半无限，并且所得到的核学习问题必须由昂贵的SDP求解器来解决。 最后，为了构造X的低维表示？ S和X？ T，所获得的K必须由PCA进一步后处理。 这可能会丢弃K中的有用信息。在本文中，我们提出了一种基于核特征提取的非线性映射φ的有效方法。 它避免了SDP的使用，从而避免了高计算负担。 此外，学习的核k可以直接推广到样本外模式。 此外，我们提出了一种利用显式低秩表示的单一内核学习方法，而不是像在MMDE中那样使用两步方法。**

First, recall that the kernel matrix K in (4) can be decomposed as K =(KK−1/2)(K−1/2K), which is often known as the empirical kernel map [Sch¨olkopf et al., 1998]. Consider the use of a (n1 + n2) × m matrix W to transform the corresponding feature vectors to a m-dimensional space. In general, m << n1 + n2. The resultant kernel matrix1 is then 

**首先，回想一下（4）中的核矩阵K可以分解为K =（KK-1/2）（K-1 / 2K），这通常被称为经验核映射[Sch¨olkopf等，1998]。 考虑使用（n1 + n2）×m矩阵？ W将相应的特征向量变换为m维空间。 通常，m << n1 + n2。 然后得到的内核矩阵1**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_4.jpg)

where W = K^(−1/2)W ∈ R^((n1+n2)×m). In particular, the corresponding kernel evaluation of k between any two patterns xi and xj is given by 

**其中W = K ^（ - 1/2）W∈R^（（n1 + n2）×m）。 特别地，任何两个模式xi和xj之间的k的相应内核评估由下式给出**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_5.jpg)

where kx =[ k(x1,x),...,k(xn1+n2,x)]^T ∈ Rn1+n2. Hence, the kernel k in (6) facilitates a readily parametricform for out-of-sample kernel evaluations. Moreover, using the deﬁnition of  K in (5), the distance between the empirical means of the two domains can be rewritten as: 

Moreover, using the deﬁnition of  K in (5), the distance between the empirical means of the two domains can be rewritten as: 

**其中kx = [k（x1，x），...，k（xn1 + n2，x）] ^T∈Rn1+ n2。 因此，（6）中的核k有助于用于样本外核评估的容易参数化形式。 此外，使用（5）中K的定义，两个域的经验平均值之间的距离可以改写为：**

**此外，使用（5）中K的定义，两个域的经验平均值之间的距离可以改写为：**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_6.jpg)

### Transfer Components Extraction  转移组份提取

In minimizing criterion (7), a regularization term tr(W^TW) is usually needed to control the complexity of W. As will be shown later in this section, this regularization term can avoid the rank deﬁciency of the denominator in the generalized eigendecomposition. The kernel learning problem for domain adaptation then reduces to: 

**在最小化准则（7）中，通常需要正则化项tr（W ^ TW）来控制W的复杂度。如本节稍后所示，该正则化项可以避免广义特征分解中分母的秩缺陷。。 域适应的内核学习问题然后简化为：**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_7.jpg)

中间公式繁多所以直接贴图





### Computational Issues  计算问题

The kernel learning algorithm in [Pan et al., 2008] relies on SDPs. As there are O((n1 +n2)2) variables in  K, the overall training complexity is O((n1 + n2)6.5) [Nesterov and Nemirovskii, 1994]. This becomes computationally prohibitive even for small-sized problems. Note that criterion (3) in this kernel learning problem is similar to the recently proposed supervised dimensionality reduction method colored MVU [Song et al., 2008], in which low-rank approximation is used to reduce the number of constraints and variables in the SDP. However, gradientdescentis requiredto reﬁnethe embedding space and thus the solution can get stuck in a local minimum. On the other hand, our proposed kernel learning method requires only a simple and efﬁcient eigendecomposition. This takes only O(m(n1 +n2)2) time when m non-zero eigenvectors are to be extracted [Sorensen, 1996].

**[Pan et al。，2008]中的内核学习算法依赖于SDP。 因为有（（n1 + n2）2）个变量？ K，整体训练复杂度为O（（n1 + n2）6.5）[Nesterov和Nemirovskii，1994]。 即使对于小型问题，这在计算上也是禁止的。 注意，该核学习问题中的标准（3）类似于最近提出的有色MVU的监督维数降低方法[Song et al。，2008]，其中使用低秩近似来减少约束和变量的数量。SDP。 但是，需要使用gradientdescent来重新嵌入空间，因此解决方案可能会陷入局部最小值。 另一方面，我们提出的内核学习方法只需要一个简单而有效的特征分解。 当要提取m个非零特征向量时，这仅需要O（m（n1 + n2）2）时间[Sorensen，1996]。**

## 相关工作

Domain adaptation, which can be considered as a special setting of transfer learning [Pan and Yang, 2008], has been widely studied in natural language processing (NLP) [Ando and Zhang, 2005; Blitzer et al., 2006; Daum´e III, 2007]. Ando and Zhang [2005] and Blitzer [2006] proposed structural correspondence learning (SCL) algorithms to learn the common feature representation across domains based on some heuristic selection of pivot features. Daum´e III [2007] designed a heuristic kernel to augment features for solving some speciﬁc domain adaptation problems in NLP. Besides, domain adaptation has also been investigated in other application areas such as sentiment classiﬁcation [Blitzer et al., 2007]. Theoretical analysis of domain adaptation has also been studied in [Ben-David et al., 2007]. 

The problem of sample selection bias (also referred to as co-variate shift) is also related to domain adaption. In sample selection bias, the basic assumption is that the sampling processes between the training data Xtrn and test data Xtst may be different. As a result, P(Xtrn) = P(Xtst), but P(Ytrn|Xtrn)=P(Ytst|Xtst). Instance re-weighting is a major technique for correcting sample selection bias [Huang et al., 2007; Sugiyama et al., 2008]. Recently, a state-ofart method, called kernel mean matching (KMM), is proposed [Huang et al., 2007]. It re-weights instances in a RKHS based on the MMD theory, which is different from our proposed method. Sugiyama et al.[2008] proposed another re-weighting algorithm, Kullback-Leibler Importance Estimation Procedure (KLIEP), which is integrated with crossvalidation to perform model selection automatically. Xing et al.[2007] proposed to correct the labels predicted by a shiftunaware classiﬁer towards a target distribution based on the mixture distribution of the training and test data. Matching distributions by re-weighting instances is also used successfully in Multi-task Learning [Bickel et al., 2008]. However, unlike instance re-weighting, the proposed TCA method can cope with noisy features (as in image data and WiFi data) by effectively denoising and ﬁnding a latent space for matching distributions across different domains simultaneously. Thus, TCA can be treated as an integration of unsupervised feature extraction and distribution matching in a latent space.

**领域适应，可以被认为是转移学习的特殊环境[Pan and Yang，2008]，已经在自然语言处理（NLP）中得到了广泛的研究[Ando和Zhang，2005; Blitzer等，2006; Daum'e III，2007]。 Ando和Zhang [2005]以及Blitzer [2006]提出了结构对应学习（SCL）算法，以基于对枢轴特征的一些启发式选择来学习跨域的共同特征表示。 Daum'e III [2007]设计了一个启发式内核来增强功能，以解决NLP中的一些特定领域适应问题。此外，领域适应也在其他应用领域进行了研究，如情绪分类[Blitzer et al。，2007]。**

**在[Ben-David等人，2007]中也研究了域适应的理论分析。样本选择偏差的问题（也称为协变量偏移）也与域自适应有关。在样本选择偏差中，基本假设是训练数据Xtrn和测试数据Xtst之间的采样过程可能不同。结果，P（Xtrn）α= P（Xtst），但P（Ytrn | Xtrn）= P（Ytst | Xtst）。实例重新加权是纠正样本选择偏差的主要技术[Huang et al。，2007; Sugiyama等，2008]。最近，提出了一种称为核平均匹配（KMM）的状态方法[Huang et al。，2007]。它基于MMD理论对RKHS中的实例进行重新加权，这与我们提出的方法不同。 Sugiyama等人[2008]提出了另一种重加权算法，Kullback-Leibler重要性估计程序（KLIEP），它与交叉验证相结合，自动执行模型选择。 Xing等人[2007]建议根据训练和测试数据的混合分布，将shiftunaware分类器预测的标签更正为目标分布。通过重新加权实例匹配分布也成功地用于多任务学习[Bickel et al。，2008]。然而，与实例重新加权不同，所提出的TCA方法可以通过有效地去噪和发现用于同时跨不同域匹配分布的潜在空间来处理噪声特征（如在图像数据和WiFi数据中）。因此，TCA可以被视为潜在空间中无监督特征提取和分布匹配的集成。**

## 实验

In this section, we apply the proposed domain adaptation algorithm TCA on two real-world problems: indoor WiFi localization and text classiﬁcation.

**在本节中，我们将提出的域自适应算法TCA应用于两个现实问题：室内WiFi定位和文本分类。**

### Cross-domain WiFi Localization 

For cross-domain WiFi localization, we use a dataset published in the 2007 IEEE ICDM Contest [Yang et al., 2008]. This dataset contains some labeled WiFi data collected in time period A (the source domain) and a large amount of unlabeled WiFi data collected in time period B (the target domain). Here, a label means the corresponding location where the WiFi data are received. WiFi data collected from different time periods are considered as different domains. The task is to predict the labels of the WiFi data collected in time period B. More speciﬁcally, all the WiFi data are collected in an indoor building around 145.5×37.5 m2, 621 labeled data are collected in time period A and 3128 unlabeled data are collected in time period B.

 We conduct a series of experiments to compare TCA with some baselines, including other feature extraction methods such as KPCA, sample selection bias (or co-variate shift) methods, KMM and KLIEP and a domain adaptation method, SCL. For each experiment, all labeled data in the source domain and some unlabeled data in the target domain are used for training. Evaluation is then performed on the remaining unlabeled data (out-of-sample) in the target domain. This is repeated 10 times and the average performance is used to measure the generalization abilities of the methods. In addition, to compare the performance between TCA and MMDE, we conduct some experiments in the transductive setting [Nigam et al., 2000]. The evaluation criterion is the Average Error Distance (AED) on the test data, and the lower the better. For determining parameters for each method, we randomly select a very small subset of the target domain data to tune parameters. The values of parameters are ﬁxed for all the experiments.

 Figure 1(a) compares the performance of Regularized Least Square Regression (RLSR) model on different feature representations learned by TCA, KPCA and SCL, and different re-weighted instances learned by KMM and KLIEP. Here, we use μ =0 .1 for TCA and the Laplacian kernel. As can be seen, the performance can be improved with the new feature representations of TCA and KPCA. TCA can achieve much higher performance because it aims at ﬁnding the leading components that minimize the differencebetween domains. Then, fromthespace spannedbythese components, the model trained in one domain can be used to perform accurate prediction in the other domain. 

**对于跨域WiFi本地化，我们使用2007年IEEE ICDM竞赛[Yang et al。，2008]中发布的数据集。该数据集包含在时间段A（源域）中收集的一些标记的WiFi数据和在时间段B（目标域）中收集的大量未标记的WiFi数据。这里，标签表示接收WiFi数据的相应位置。从不同时间段收集的WiFi数据被视为不同的域。任务是预测在时间段B中收集的WiFi数据的标签。更具体地，所有WiFi数据被收集在145.5×37.5m2的室内建筑物中，在时间段A中收集621个标记数据并且3128个未标记数据是在时间段B收集**

 **我们进行了一系列实验来比较TCA与一些基线，包括其他特征提取方法，如KPCA，样本选择偏差（或协变量移位）方法，KMM和KLIEP以及域适应方法SCL。对于每个实验，源域中的所有标记数据和目标域中的一些未标记数据用于训练。然后对目标域中剩余的未标记数据（样本外）进行评估。重复10次，平均性能用于测量方法的泛化能力。此外，为了比较TCA和MMDE之间的性能，我们在转导环境中进行了一些实验[Nigam等，2000]。评估标准是测试数据的平均误差距离（AED），越低越好。为了确定每种方法的参数，我们随机选择目标域数据的一个非常小的子集来调整参数。所有实验都固定了参数值。**

 **图1（a）比较了正则化最小二乘回归（RLSR）模型对TCA，KPCA和SCL学习的不同特征表示的性能，以及KMM和KLIEP学习的不同重新加权实例。这里，我们使用μ= 0.1为TCA和拉普拉斯算子核。可以看出，使用TCA和KPCA的新功能表示可以提高性能。 TCA可以实现更高的性能，因为它旨在找到最小化域之间差异的主要组件。然后，从这些组件跨越的空间，在一个域中训练的模型可用于在另一个域中执行准确预测。**

Figure 1(b) shows the results under a varying number of unlabeled data in the target main. As can be seen, with only a few unlabeled data in the target domain, TCA can still ﬁnd a good feature representation to bridge between domains. 

Since MMDE cannot generalize to out-of-sample patterns, in order to compare TCA with MMDE, we conduct another series of experiments in a transductive setting, which means that the trained models are only evaluated on the unlabeled data that are used for learning the latent space. In Figure 1(c), we apply MMDE and TCA on 621 labeled data from the source domain and 300 unlabeled data from the target domain to learn new representations, respectively, and then train RLSR on them. More comparison results in terms of ACE with varying number of training data are shown in Table 1. The experimental results show that TCA is slightly higher (worse) than MMDE in terms of AED. This is due to the nonparametric kernel matrix learned by MMDE, which can ﬁt the observed unlabeled data better. However, as mentioned in Section 3.4, the cost of MMDE is expensive due to the computationally intensive SDP. The comparison results between TCA and MMDE in terms of computational time on the WiFi dataset are shown in Table 2.

**图1（b）显示了目标主数据中不同数量的未标记数据下的结果。可以看出，由于目标域中只有少量未标记的数据，TCA仍然可以找到一个良好的特征表示来桥接域之间。**

**由于MMDE无法推广到样本外模式，为了将TCA与MMDE进行比较，我们在转换设置中进行了另一系列实验，这意味着训练模型仅针对用于学习的未标记数据进行评估。潜在的空间。在图1（c）中，我们对来自源域的621个标记数据和来自目标域的300个未标记数据应用MMDE和TCA以分别学习新表示，然后在它们上训练RLSR。表1中显示了具有不同训练数据的ACE的更多比较结果。实验结果表明，就AED而言，TCA略高于（差）MMDE。这是由于MMDE学习的非参数核矩阵，可以更好地处理观察到的未标记数据。但是，如3.4节所述，由于计算密集型SDP，MMDE的成本很高。在WiFi数据集上的计算时间方面，TCA和MMDE之间的比较结果如表2所示。**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_11.jpg)

					图1.平均误差距离的比较（以m为单位）

Table 1: ACE (in m) of MMDE and TCA with 10 dimensions and varying # training data (# labeled data in the source domain is ﬁxed to 621, # unlabeled data in the target domain varies from 100 to 800.) 

**表1：具有10个维度和不同＃训练数据的MMDE和TCA的ACE（以m为单位）（源域中的＃标记数据固定为621，目标域中的＃未标记数据在100到800之间变化。）**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_12.jpg)

Table 2: CPU training time (in sec) of MMDE and TCA with varying # training data.

**表2：具有不同＃训练数据的MMDE和TCA的CPU训练时间（以秒为单位）。**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_13.jpg)

### Cross-domain Text Classiﬁcation 跨域文本分类

In this section, we perform cross-domain binary classiﬁcation experiments on a preprocessed dataset of Reuters-21578. These data are categorized to a hierarchical structure. Data from different sub-categories under the same parent category are considered to be from different but related domains. The task is to predict the labels of the parent category. By following this strategy, three datasets orgs vs people, orgs vs places and people vs places are constructed. We randomly select 50% labeled data from the source domain, and 35% unlabeled data from the target domain. Evaluation is based on the (out-of-sample) testing of the remaining 65% unlabeled data in the target domain. This is repeated 10 times and the average results reported. 

Similar to the experimental setting on WiFi localization, we conduct a series of experiments to compare TCA with KPCA, KMM, KLIEP and SCL. Here, the supportvector machine (SVM) is used as the classiﬁer. The evaluation criterion is the classiﬁcation accuracy (the higher the better). We experiment with both the RBF kernel and linear kernel for feature extraction or re-weighting used by KPCA, TCA and KMM. The kernel used in the SVM for ﬁnal prediction is a linear kernel, and the parameter μ in TCA is set to 0.1. 

As can be seen from Table 3, different from experiments on the WiFi data, sample selection bias methods, such as KMM and KLIEP perform better than KPCA and PCA on the text data. However, with the feature presentations learned by TCA, SVM performs the best for cross-domain classiﬁcation. This is because TCA not only discovers latent topics behind the text data, but also matches distributions across domains in the latent space spanned by the latent topics. Moreover, the performance of TCA using the RBF kernel is more stable.

**在本节中，我们对Reuters-21578的预处理数据集执行跨域二进制分类实验。这些数据被分类为分层结构。来自相同父类别下的不同子类别的数据被认为来自不同但相关的域。任务是预测父类别的标签。通过遵循这一策略，构建了三个数据集orgs vs people，orgs vs places和people vs places。我们从源域中随机选择50％标记的数据，从目标域中随机选择35％未标记的数据。评估基于目标域中剩余65％未标记数据的（样本外）测试。重复10次，报告平均结果。**

**与WiFi定位的实验设置类似，我们进行了一系列实验来比较TCA与KPCA，KMM，KLIEP和SCL。这里，支持向量机（SVM）用作分类器。评估标准是分类准确度（越高越好）。我们尝试使用RBF内核和线性内核进行KPCA，TCA和KMM使用的特征提取或重新加权。 SVM中用于最终预测的内核是线性内核，TCA中的参数μ设置为0.1。**

**从表3可以看出，与WiFi数据的实验不同，样本选择偏差方法，例如KMM和KLIEP在文本数据上的表现优于KPCA和PCA。但是，通过TCA学习的功能演示，SVM可以最好地进行跨域分类。这是因为TCA不仅发现了文本数据背后的潜在主题，而且还匹配潜在主题所跨越的潜在空间中的域之间的分布。而且，使用RBF内核的TCA性能更稳定。**

![im](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/TCA_14.jpg)

## Conclusion and Future Work 结论和未来的工作

Learning feature representations is of primarily an important task for domain adaptation. In this paper, we propose a new feature extraction method, called Transfer Component Analysis (TCA), to learn a set of transfer components which reduce the distance across domains in a RKHS. Compared to the previously proposed MMDE for the same task, TCA is much more efﬁcient and can be generalized to out-of-sample patterns. Experiments on two real-world datasets verify the effectiveness of the proposed method. In the future, we are planning to take side information into account when learning the transfer components across domains, which may be better for the ﬁnal classiﬁcation or regression tasks.

**学习特征表示主要是域适应的重要任务。 在本文中，我们提出了一种新的特征提取方法，称为传递分量分析（TCA），以学习一组传递组件，这些组件减少了RKHS中跨域的距离。 与先前提出的用于相同任务的MMDE相比，TCA更有效，并且可以推广到样本外模式。 对两个真实数据集的实验验证了所提方法的有效性。 将来，我们计划在跨域学习传输组件时考虑辅助信息，这对于最终的分类或回归任务可能更好。**

------



