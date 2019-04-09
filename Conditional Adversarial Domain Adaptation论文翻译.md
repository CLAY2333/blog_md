---
title: Conditional Adversarial Domain Adaptation论文翻译
date: 2018-11-26 21:28:48
categories: 论文翻译
tags: [论文翻译,机器学习,深度学习,迁移学习]
---

<Excerpt in index | 首页摘要> 

本文为NIPS2018的一篇关于域适应的文章，在2015年的文章上对于只对特征进行适应上提出了对类别也进行适应的论文。

<!-- more -->

<The rest of contents | 余下全文>

# Conditional Adversarial Domain Adaptation

## 摘要

Adversarial learning has been embedded into deep networks to learn transferable representations for domain adaptation. Existing adversarial domain adaptation methods may struggle to align different domains of multimode distributions that are native in classification problems. In this paper, we present conditional adversarial domain adaptation, a novel framework that conditions the adversarial adaptation models on discriminative information conveyed in the classifier predictions. Conditional domain adversarial networks are proposed to enable discriminative adversarial adaptation of multimode domains. The experiments testify that the proposed approaches exceed the state-of-the-art performance on three domain adaptation datasets.

**对抗性学习已嵌入到深层网络中，以学习域适应的可转移表示。 现有的对抗域自适应方法可能难以对齐分类问题中原生的多模分布的不同域。 在本文中，我们提出了条件对抗域适应，这是一个新的框架，它对分类器预测中传达的判别信息的对抗适应模型进行了调整。 提出条件域对抗性网络以实现多模域的辨别性对抗性适应。 实验证明，所提出的方法超过了三个领域适应数据集的最新性能。**

## 引言

Deep networks have significantly improved the state of the arts for diverse machine learning problems and applications. When trained on large-scale datasets, deep networks learn representations which are generically useful across a variety of tasks and domains. Unfortunately, the huge performance gains come only when massive amounts of labeled data are available for supervised learning. Since manual labeling of sufficient training data for diverse application domains is often prohibitive, for a target task short of labeled data, there is strong motivation to build effective learners that can leverage rich labeled data from a different source domain. However, this learning paradigm suffers from the shift in data distributions across different domains, which poses an obstacle for adapting learning models to unlabeled target tasks (Quionero-Candela et al., 2009; Pan & Yang, 2010).

**深度网络显着改善了各种机器学习问题和应用的艺术状态。 在大规模数据集上接受培训时，深层网络会学习在各种任务和领域中通用的表示。 不幸的是，只有当大量标记数据可用于监督学习时，才能获得巨大的性能提升。 由于针对不同应用领域的足够训练数据的手动标记通常是令人望而却步的，因此对于没有标记数据的目标任务，存在建立有效学习者的强烈动机，其可以利用来自不同源域的丰富标记数据。 然而，这种学习范式受到不同领域数据分布的转变的影响，这使得学习模型适应未标记的目标任务成为障碍。**

Learning a discriminative model that reduces the dataset shift between training and testing distributions is known as transfer learning or domain adaptation (Pan & Yang, 2010). Previous domain adaptation methods in the shallow regime either bridge the source and target by learning invariant feature representations or estimating instance importance using labeled source data and unlabeled target data (Huang et al., 2006; Pan et al., 2011; Gong et al., 2013). Recent advances of deep domain adaptation methods leverage deep networks to learn transferable representations by embedding domain adaptation modules in deep architectures, which can simultaneously disentangle the explanatory factors of variations behind data and match feature distributions across domains (Ganin & Lempitsky, 2015; Ganin et al., 2016; Long et al., 2015; 2016; 2017; Tzeng et al., 2015; 2017).

**学习一种减少训练和测试分布之间数据集转换的判别模型称为转移学习或领域适应。 浅层区域中的先前域适应方法通过学习不变特征表示或使用标记的源数据和未标记的目标数据估计实例重要性来桥接源和目标。 深度域自适应方法的最新进展利用深度网络通过在深层体系结构中嵌入域适配模块来学习可转移表示，这可以同时解开数据背后变异的解释因素并匹配跨域的特征分布。**

In particular, adversarial domain adaptation methods (Ganin & Lempitsky, 2015; Tzeng et al., 2015; 2017) are among the top-performing deep architectures for domain adaptation. These methods work in a two-player game similarly to the generative adversarial networks (Goodfellow et al., 2014). A domain discriminator, e.g. MLP, is learned by minimizing the classification error of distinguishing the source from the target domains, while a deep classification model, e.g. CNN, learns transferable representations that are indistinguishable to confuse the domain discriminator. Despite their efficacy, existing adversarial domain adaptation methods may be confined by two key bottlenecks. First, when data distributions embody complex multimode structures, adversarial domain adaptation may fail to capture such multimode structures to ensure fine-grained alignment of distributions, known as the mode collapse difficulty in generative adversarial networks (Goodfellow et al., 2014; Che et al., 2017). Second, the dataset shifts linger in domain-specific feature and classifier layers (Yosinski et al., 2014), and adversarial adaptation of a particular layer is not sufficient to bridge the domain shifts.

**特别是，对抗域适应方法是领域适应的最佳表现深层架构。与生成性对抗网络类似，这些方法在双人游戏中起作用。域鉴别符，例如通过最小化区分源和目标域的分类误差来学习MLP，同时深度分类模型，例如，卷积神经网络（CNN）学习可混淆的表述，这些表示无法区分混淆域鉴别器。尽管它们有效，但现有的对抗域适应方法可能受到两个关键瓶颈的限制。首先，当数据分布体现复杂的多模结构时，对抗域适应可能无法捕获这种多模结构以确保分布的细粒度对齐，称为生成对抗网络中的模式崩溃困难。其次，数据集在特定领域的特征和分类器层中移位，并且特定层的对抗性适应不足以桥接域移位。**

In this paper, we tackle the two aforementioned challenges by formalizing a conditional adversarial domain adaptation framework. Recent advances in the conditional generative adversarial networks (Mirza & Osindero, 2014; Odena et al., 2017) disclose that the mode collapse can be alleviated by conditioning the generator and discriminator on relevant discriminative information. Motivated by the conditioning insight, this paper presents Conditional Domain Adversarial Networks (CDAN) to exploit discriminative information conveyed in the classifier predictions to inform adversarial adaptation. The key to the CDAN models is a novel conditional domain discriminator conditioned on the multilinear map of domain-specific feature representations and classifier predictions. A well-specified Focal Loss is devised to focus the discriminator on hard-to-match examples, potentially alleviating the gradient vanishing problem. The overall system can be solved in linear-time through back-propagation. Extensive experiments demonstrate that our models exceed state of the art results on three domain adaptation datasets.

**在本文中，我们通过形式化条件对抗域适应框架来解决上述两个挑战。条件生成对抗网络的最新进展公开了通过调节相关判别信息的生成器和鉴别器可以减轻模式崩溃。受条件洞察力的启发，本文提出了条件域对抗网络（CDAN）来利用分类器预测中传达的判别信息来通知对抗性适应。 CDAN模型的关键是以区域特定的特征表示和分类器预测的多线性映射为条件的新型条件域鉴别器。设计了明确的焦点损失，将鉴别器集中在难以匹配的例子上，可能减轻梯度消失问题。整个系统可以通过反向传播在线性时间内求解。大量实验表明，我们的模型超过了三个领域适应数据集的最新结果。**

## 相关工作

Domain adaptation (Pan & Yang, 2010; Quionero-Candela et al., 2009) generalizes a learner across different domains of different distributions (Sugiyama et al., 2008; Pan et al., 2011; Duan et al., 2012; Gong et al., 2013; Zhang et al., 2013). It finds wide applications in computer vision (Saenko et al., 2010; Gopalan et al., 2011; Gong et al., 2012; Hoffman et al., 2014) and natural language processing (Collobert et al., 2011; Glorot et al., 2011). Recent studies reveal that deep networks learn more transferable representations that disentangle the explanatory factors of variations behind data (Bengio et al., 2013) and manifest invariant factors underlying different populations (Glorot et al., 2011; Oquab et al., 2013). As deep representations can only reduce, but not remove, the cross-domain discrepancy (Yosinski et al., 2014), recent research on deep domain adaptation further embeds domain-adaptation modules in deep networks to boost transfer performance (Tzeng et al., 2014; Ganin & Lempitsky, 2015; Tzeng et al., 2015; Long et al., 2015; 2016; 2017).

Adversarial learning has been successfully explored for generative modeling. Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) constitute two networks in a twoplayer game: a generator that captures data distribution and a discriminator that distinguishes between generated samples and real data. The networks are trained in a mini-max fashion such that the generator is learned to fool the discriminator. Several difficulties of GANs have been addressed, e.g. improved training (Arjovsky et al., 2017; Arjovsky & Bottou, 2017), mode collapse (Mirza & Osindero, 2014; Che et al., 2017; Metz et al., 2017; Odena et al., 2017). These technologies have not been leveraged to domain adaptation.

**域适应将学习者推广到不同分布的不同领域。它在计算机视觉中得到广泛应用和自然语言处理。最近的研究表明，深层网络学习了更多可转移的表示，它们解开了数据背后变异的解释因素，并显示了不同人群背后的不变因素。由于深度表示只能减少但不能消除跨域差异，最近对深域适应的研究进一步将域适配模块嵌入到深层网络中以提高传输性能。**

**已经成功地探索了对抗性学习的生成建模。生成性对抗网络（GAN）构成了双层游戏中的两个网络：捕获数据分布的生成器和区分生成的样本和实际数据的鉴别器。以迷你最大化方式训练网络，以便学习发生器以欺骗鉴别器。已经解决了GAN的几个困难，例如改进的培训，模式崩溃。这些技术尚未用于域适应。**

## 有条件的对抗性适应

In unsupervised domain adaptation, we are given a source domain D labeled examples and a target domain D unlabeled examples.The source domain and target domain are sampled from joint distributions P and Q respectively, while the IID assumption is violated as P 6= Q. The goal of this paper is to design a deep network y = G(x) which formally reduces the shifts in the joint distributions across domains, such that the target risk G can be minimized by jointly minimizing the source risk  G and the cross-domain discrepancy D with conditional adversarial learning

**在无监督域自适应中，给出源域D标记的示例和目标域D未标记的示例。源域和目标域分别从关节分布P和Q采样，而IID假设被违反为P 6 = Q. 本文的目标是设计一个深度网络y = G（x），它正式减少跨域的联合分布的变化，从而通过联合最小化源风险G和跨域来最小化目标风险G. 差异D与条件对抗性学习**

### Domain Adversarial Neural Network  域对抗神经网络

Two key properties prosper the deep learning applications : discriminability and transferability. Deep networks can learn distributed representations that are both discriminative to source tasks at hand, and transferable to target tasks of interest . The favorable transferability of deep representations leads to some representative deep transfer learning methods . These methods exploit the domain adaptation theory : controlling the target risk by jointly bounding the source risk and the domain discrepancy underlying data representations.\

**两个关键属性使深度学习应用繁荣：可辨性和可转移性。 深度网络可以学习分布式表示，这些表示既可以区分手头的源任务，也可以转移到感兴趣的目标任务。 深度表征的有利可转移性导致一些代表性的深度转移学习方法。 这些方法利用域适应理论：通过共同限制源风险和基础数据表示的域差异来控制目标风险。**

Adversarial learning, the core idea to enabling generative adversarial networks (GANs), has been successfully explored to adversarial domain adaptation approaches . Denote by f = F (x) the feature representation and by g = G(x) the classifier prediction generated with the deep network G. The domain adversarial learning procedure is a two-player game, where the first player is the domain discriminator D trained to distinguish the source domain from the target domain and the second player is the feature representation F trained simultaneously to confuse the domain discriminator.

**对抗性学习是实现生成性对抗性网络（GAN）的核心思想，已经成功地探索了对抗性领域适应方法。 用f = F（x）表示特征表示，用g = G（x）表示用深网络G生成的分类器预测。域对抗学习过程是双人游戏，其中第一个玩家是域鉴别器D 训练以区分源域和目标域，第二个播放器是同时训练的特征表示F以混淆域鉴别器。**

In domain adversarial neural network , the domain discriminator D is learned by minimizing the error E0(D,G) over the source and target domains, and the domain-invariant representation f = F(x) is learned by maximizing the error E0(D,G) of the domain discriminator D. Simultaneously, the error E(G) of the label classifier g = G(x) is minimized to guarantee lower source risk. The error functions of D and G are formulated respectively as

**在域对抗神经网络中，通过最小化源域和目标域上的误差E0（D，G）来学习域鉴别器D，并且通过最大化误差E0（D）来学习域不变表示f = F（x）。 域鉴别器D，G）同时，标签分类器g = G（x）的误差E（G）被最小化以保证较低的源风险。 D和G的误差函数分别表示为**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_3.png)

Figure 1. The architectures of Conditional Domain Adversarial Networks (CDAN) for deep domain adaptation, where domain-specific feature representation f and classifier prediction g embody the cross-domain gap and should be adapted jointly by domain discriminator D, the Focal Loss (FL) is proposed to focus D on hard-to-match examples. (a) Multilinear (M) conditioning, applicable to lower-dimensional scenario, where D is conditioned on classifier prediction g via multilinear map f ⊗g; (b) Randomized Multilinear (RM) conditioning, fit to higher-dimensional scenario, where D is conditioned on classifier prediction g via randomized multilinear map  (Rff)  (Rgg).

**图1.用于深度域自适应的条件域对抗网络（CDAN）的体系结构，其中特定于域的特征表示f和分类器预测g体现了跨域间隙，并且应由域鉴别器D（Focal Loss）共同调整（FL） 建议将D专注于难以匹配的例子。 （a）多线性（M）条件反射，适用于低维场景，其中D通过多线性映射f⊗g以分类器预测g为条件; （b）随机多线性（RM）调节，适合于更高维度的场景，其中D通过随机多线性映射（Rff）（Rgg）以分类器预测g为条件。**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_1.png)

where L(·,·) is the cross-entropy loss, fis and   are the feature representations of source and target data  , F is a subnetwork of G and thus is optimized within G. The two-player game of Domain Adversarial Neural Network is where λ is a trade-off parameter between the two objectives that shape the feature representations to be domain-invariant during domain adversarial training of deep neural networks. It has been shown by (Ganin & Lempitsky, 2015) that the the error function of domain discriminator is well corresponded where only the feature distributions P(f) and Q(f) change across domains.

**其中L（·，·）是交叉熵损失，fis是源数据和目标数据的特征表示，F是G的子网，因此在G内进行优化。域对抗神经网络的双人游戏是 其中λ是两个目标之间的权衡参数，这两个目标在深度神经网络的域对抗训练期间将特征表示形成为域不变的。 （Ganin＆Lempitsky，2015）已经表明，域鉴别器的误差函数很好地对应于只有特征分布P（f）和Q（f）跨域变化的情况。**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_2.png)

### Conditional Domain Adversarial Network 条件域对抗网络

In principle, existing adversarial domain adaptation methods can be further strengthened in two indispensable directions. First, when the joint distributions of feature and label, P(xs,ys) and Q(xt,yt), are non-identical across domains, adapting only the feature representation f to make it domain-invariant may be insufficient. Due to a quantitative study , deep representations eventually transition from general to specific along deep networks, with transferability decreased remarkably in the domain-specific feature layer f and classifier layer g. In other words, the joint distributions of feature representation f and classifier prediction g should still be non-identical in these domain adversarial networks. Second, when the feature distribution is multimode, which is real scenario due to the nature of multi-class classification, adapting only the feature representation may be challenging for adversarial networks. Recent researches reveal the mode collapse challenge, highlighting the risk of failure in matching a fraction of modes underlying different distributions with adversarial networks. Namely, even if the discriminator is fully confused, we have no theoretical guarantee that two different distributions are made identical .

**原则上，现有的对抗域适应方法可以在两个不可或缺的方向上得到进一步加强。首先，当特征和标签的联合分布P（xs，ys）和Q（xt，yt）在域之间不相同时，仅调整特征表示f以使其域不变可能是不够的。由于定量研究，深度表示最终沿着深度网络从一般过渡到特定，在特定于域的特征层f和分类器层g中可转移性显着降低。换句话说，在这些域对抗性网络中，特征表示f和分类器预测g的联合分布应该仍然是不相同的。其次，当特征分布是多模式时，由于多类别分类的性质而是真实场景，仅适应特征表示对于对抗性网络可能是具有挑战性的。最近的研究揭示了模式崩溃的挑战，突出了将不同分布的一部分模式与对抗性网络相匹配的失败风险。也就是说，即使鉴别器完全混淆，我们也没有理论上保证两个不同的分布是相同的。**

In this paper, we tackle the two aforementioned challenges by formalizing a conditional adversarial domain adaptation framework. Recent advances in the conditional generative adversarial networks (CGANs) disclose that the mode collapse challenge can be alleviated by conditioning the generator and discriminator on relevant information, such associated labels and affiliated modality. CGANs and variants generate globally coherent and high resolution samples from datasets with high variability and multimode distributions. Motivated by the philosophy of conditional GANs, we observe that in adversarial domain adaptation, the classifier prediction g conveys rich discriminative information that potentially reveals the multimode structures, which can be conditioned on when adapting the feature representation f. By conditioning, domain variances in feature representation f and classifier prediction g can be modeled simultaneously.

**在本文中，我们通过形式化条件对抗域适应框架来解决上述两个挑战。 条件生成对抗网络（CGAN）的最新进展公开了通过调节相关信息（例如相关标签和附属模态）的生成器和鉴别器可以减轻模式崩溃挑战。 CGAN和变体从具有高可变性和多模分布的数据集生成全局相干和高分辨率样本。 在条件GAN的哲学的推动下，我们观察到在对抗域自适应中，分类器预测g传达了可能揭示多模结构的丰富的判别信息，其可以在调整特征表示f时受到条件限制。 通过调节，可以同时对特征表示f和分类器预测g中的域变化进行建模。**

For notation clarity, denote by h = (f,g) the joint variable of both feature representation and classifier prediction. We enable the conditioning of domain discriminator D on the classifier prediction g by reformulating the discriminator as

**对于符号清晰度，通过h =（f，g）表示特征表示和分类器预测的联合变量。 我们通过将鉴别器重新表示为分区器预测g来启用域鉴别器D的调节**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_4.png)

This conditional domain discriminator can potentially tackle the two aforementioned challenges of adversarial domain adaptation. In spite of its potential power and simple form, D(f,g) should be formulated by well-specified functional. A simplest functional is D([f,g]), where we concatenate the feature representation and classifier prediction in a single vector [f,g] that is fed to conditional domain discriminator D. This conditioning strategy is widely adopted by existing conditional GANs . However, with the concatenation strategy, f and g are independent on each other, thus failing to fully capture multiplicative interactions between feature representation and classifier prediction, which are crucial to domain adaptation. As a result, the multimode information conveyed in classifier prediction cannot be fully exploited to tackle the challenges of mode collapse and joint variability.

**该条件域鉴别器可以潜在地解决上述两种对抗域适应的挑战。尽管它具有潜在的力量和简单的形式，但D（f，g）应该通过明确的功能来制定。最简单的函数是D（[f，g]），其中我们在单个向量[f，g]中连接特征表示和分类器预测，该向量[f，g]被馈送到条件域鉴别器D.这种调节策略被现有的条件GAN广泛采用。 。然而，利用级联策略，f和g彼此独立，因此无法完全捕获特征表示和分类器预测之间的乘法交互，这对于域自适应是至关重要的。结果，在分类器预测中传达的多模信息不能被充分利用以应对模式崩溃和联合可变性的挑战。Multilinear Map A.k.a., tensor product, can model the** 

multiplicative interactions between different variables. The bilinear map has been explored in Bilinear CNNs  achieving remarkable improvements for fine-grained recognition. Tensor product of infinite-dimensional nonlinear feature maps has been successfully applied to embed joint distribution or conditional distribution into reproducing kernel Hilbert spaces (RKHSs)  to enable learning from distributions, lending its great capability of modeling multiplicative interactions across different random variables.
In this paper, we condition D on g with multilinear map as

**多线性Map A.k.a.，张量积，可以模拟不同变量之间的乘法相互作用。在双线性CNN中探索了双线性映射，实现了细粒度识别的显着改进。无限维非线性特征映射的张量积成功地应用于将联合分布或条件分布嵌入到再生核希尔伯特空间（RKHS）中以使得能够从分布中学习，从而使其具有跨不同随机变量建模乘法交互的强大能力。
在本文中，我们用多线性映射条件对D进行条件计算**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_5.png)

where T⊗ is the multilinear map and D(f,g) = D(f ⊗ g).As such, the conditional domain discriminator successfully models the multimode information and joint distributions. Thanks to multi-linearity, multilinear map can be applied to random vectors with different cardinalities and magnitudes.

**其中T⊗是多线性映射，D（f，g）= D（f⊗g）。因此，条件域鉴别器成功地模拟了多模信息和联合分布。 由于多线性，多线性映射可以应用于具有不同基数和幅度的随机向量。**

A big disadvantage of the multilinear map is the dimension explosion issue. Denote by df and dg the dimensions of vectors f and g respectively, then the dimension of multilinear map f ⊗ g will be df × dg, usually too high-dimensional to be embedded into deep networks without causing parameter explosion. This paper addresses the dimension explosion of multilinear map by randomized methods . Note that tensor product holds

**多线性映射的一大缺点是尺寸爆炸问题。 用df和dg分别表示向量f和g的维数，那么多线性映射f⊗g的维数将是df×dg，通常太高维无法嵌入深层网络而不会引起参数爆炸。 本文通过随机方法解决多线性图的尺寸爆炸问题。 请注意，张量积保持不变**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_6.png)

where h (·, ·）i is the inner-product, and T (f, g) is the explicit randomized multilinear map of dimension d << df × dg and

**其中h(.,.)中i是内积，T(f,g)是维数d << df × dg的显式随机多线程映射**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_7.png)

where  is the Hadamard product, Rf and Rg are random matrices that are sampled only once and fixed throughout learning, and each of their element Rij follows a symmetric distribution with univariance, i.e. E[Rij] = 0,E[Rij2 ] = 1. Applicable distributions include Gaussian distribution and uniform distribution. Since the inner-product on T⊗ can be accurately approximated through the inner-product on T, we can directly adopt  for computation efficiency. We guarantee the above approximation quality by a theorem.

**在Hadamard乘积中，Rf和Rg是随机矩阵，其仅被采样一次并且在整个学习期间被固定，并且它们的每个元素Rij遵循具有单变量的对称分布，即E [Rij] = 0，E [Rij2] = 1。 适用的分布包括高斯分布和均匀分布。 由于T⊗上的内积可以通过T上的内积准确逼近，我们可以直接采用计算效率。 我们通过一个定理保证上述近似质量。** 

Theorem 1. The expectation and variance of the randomized multilinear map T (f, g) (7) for T⊗ (f, g) (5) satisfy	

**定理1.随机多线性映射的期望和方差T (f, g) (7) for T⊗ (f, g) (5)满足**

where  while is computed similarly, C and C0 are constants.

**其中while类似地计算，C和C0是常量。**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_8.png)

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_9.png)

Theorem 1 verifies that T is an unbiased estimate of T⊗, and the estimation variance depends only on the fourth-order moments  and , which are constants for many symmetric distributions with univariance: (a) for normal distribution, E[(Rij)4] = 3; (b) for uniform distribution,E[(Rij)4] = 1.8. Therefore, uniform distribution yields the lowest estimation variance and best approximation quality. For easiness and simplicity, we define the final conditioning strategy used by our conditional domain discriminator D as where 4096 is the largest number of units in typical deep networks, and if dimension of multilinear map T⊗ is larger than 4096, we will opt to randomized multilinear map T.

**定理1验证T是T⊗的无偏估计，并且估计方差仅取决于四阶矩，并且是具有单不等式的许多对称分布的常数：（a）对于正态分布，E [（Rij）4 ] = 3; （b）为均匀分布，E [（Rij）4] = 1.8。 因此，均匀分布产生最低估计方差和最佳近似质量。 为了方便和简单，我们定义了条件域鉴别器D使用的最终调节策略，其中4096是典型深度网络中最大的单元数，如果多线性映射T⊗的维数大于4096，我们将选择随机化 多线性映射T.**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_10.png)

Focal Loss The minimax optimization problem (3) for domain discriminator (1) and its conditional variant (4) can be problematic, since early on during training the domain discriminator converges quickly, causing gradient vanishing difficulty. The widely-adopted cross-entropy loss will easily approaches zero for examples that are easy-to-distinguish across domains. Worse still, easy-to-distinguish examples are exactly hard-to-match examples, as their corresponding gradients are vanished and thus domain adversarial learning is suspended for them. To approach this technical difficulty, we are motivated by the recent Focal Loss (Lin et al., 2017) that focuses the model on hard-to-classify examples, which yields state-of-the-art performance for visual understanding. Different from their strategy, we apply inverse Focal Loss to our conditional domain discriminator, focusing the discriminator on easy-to-distinguish, i.e. hard-to-match, examples. Hence we change the focal strategy from a monotonically decreasing power function to a monotonically increasing exponential function. The final conditional discriminator is where exp(D(·)) is the inverse focal weight for each source example, and exp(1 − D(·)) is the inverse focal weight for each target example. As can be verified, easy-to-distinguish examples, i.e. source examples with large D(·) and target examples with large 1−D(·), will have much larger weights. Figure 2 provides an intuitive comparison of different losses. Our weighting strategy circumvents the gradient vanishing difficulty for easy-to-distinguish examples, making them better matched across domains in the adversarial procedure.

**焦点丢失域鉴别器（1）及其条件变量（4）的最小极大优化问题（3）可能是有问题的，因为在训练期间早期，域鉴别器快速收敛，导致梯度消失困难。对于易于区分跨域的示例，广泛采用的交叉熵损失很容易接近零。更糟糕的是，易于区分的例子恰好是难以匹配的例子，因为它们相应的渐变消失了，因此对它们暂停了域对抗性学习。为了解决这一技术难题，我们受到最近的Focal Loss（Lin et al。，2017）的激励，该模型将模型集中在难以分类的示例上，这些示例为视觉理解提供了最先进的性能。与他们的策略不同，我们将反焦点损失应用于我们的条件域鉴别器，将鉴别器集中在易于区分，即难以匹配的示例上。因此，我们将焦点策略从单调递减的幂函数改变为单调递增的指数函数。最终的条件鉴别器是**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_11.png)**exp（D（·））是每个源示例的反焦点权重，exp（1-D（·））是每个目标示例的反焦点权重。可以证实，易于区分的例子，即具有大D（·）的源示例和具有大1-D（·）的目标示例，将具有更大的权重。图2提供了不同损耗的直观比较。我们的加权策略避免了易于区分的示例中的梯度消失难度，使得它们在对抗性过程中更好地匹配各个域。**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_12.png)

*Figure 2. Cross-entropy loss, focal loss, and our inverse focal loss. Axis X denotes probability given by D, and Y denotes loss value.

**图2.交叉熵损失，焦点丢失和我们的反向焦点损失。 轴X表示由D给出的概率，Y表示损失值。**

Conditional Domain Adversarial Network We present conditional domain adversarial network (CDAN) for deep domain adaptation. The feature representations in the lower layers are safely transferable and can be fine-tuned to the target domain (Yosinski et al., 2014). Hence, we only perform adversarial adaptation over the domain-specific feature representation f and classifier prediction g. To enable conditional adversarial domain adaptation, we jointly minimize the label classifier error (2) with respect to the label classifier G, minimize the Focal Loss (10) with respect to conditional domain discriminator D, and maximize the Focal Loss (10) with respect to the feature extractor F that is subsumed in label classifier G. This leads to the optimization problem for learning conditional domain adversarial network (CDAN):

**条件域对抗网络我们提出了用于深度域自适应的条件域对抗网络（CDAN）。 较低层中的特征表示是安全可转移的，并且可以微调到目标域。 因此，我们仅对特定于域的特征表示f和分类器预测g执行对抗性适应。 为了启用条件对抗域适应，我们联合最小化关于标签分类器G的标签分类器误差（2），最小化关于条件域鉴别器D的焦点损失（10），并最大化焦点损失（10） 包含在标签分类器G中的特征提取器F.这导致学习条件域对抗网络（CDAN）的优化问题：**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_13.png)

where λ is a balance parameter between label classifier and conditional domain discriminator, and note that h = (f,g) is the joint variable of domain-specific feature representation f and classifier prediction g requiring adversarial adaptation. As a rule of thumb, we can safely set f as the last feature layer representation and g as the classifier layer prediction. In special cases where lower-layer features are not safely transferable, typically in pixel-level adaptation tasks (Isola et al., 2017), we may change f to other layer representations.

**其中λ是标签分类器和条件域鉴别器之间的平衡参数，并且注意h =（f，g）是特定于域的特征表示f和需要对抗适应的分类器预测g的联合变量。 根据经验，我们可以安全地将f设置为最后一个要素图层表示，将g设置为分类器层预测。 在低层特征不能安全转移的特殊情况下，通常在像素级适应任务中（Isola等，2017），我们可以将f改为其他层表示。**

## Experiments 实验

We evaluate conditional domain adversarial networks with state of the art transfer learning and deep learning methods. Codes, datasets and configurations will be released online.

我们使用最先进的转移学习和深度学习方法评估条件域对抗网络。 代码，数据集和配置将在线发布。

### 安装

Office-31  is the most widely used dataset for visual domain adaptation, comprising 4,652 images and 31 categories collected from three distinct domains: Amazon (A), with images from amazon.com, Webcam (W) and DSLR (D), with images respectively taken by web camera and digital SLR camera in different environmental settings. We evaluate all methods across three transfer tasks A → W, D → W and W → D as commonly used in  and across another three transfer tasks A → D, D → A and W → A as in . 

**Office-31是用于视觉域适应的最广泛使用的数据集，包括4,652个图像和从三个不同域收集的31个类别：亚马逊（A），来自amazon.com，网络摄像头（W）和DSLR（D）的图像，带有图像分别采用网络摄像头和数码单反相机在不同的环境设置下拍摄。我们评估三个传输任务A→W，D→W和W→D中的所有方法，如在另外三个传输任务A→D，D→A和W→A中常用的那样。**

ImageCLEF-DA  is a benchmark for ImageCLEF 2014 domain adaptation challenge, organized by selecting the 12 common categories shared by three public datasets, and each is considered as a domain: Caltech-256 (C), ImageNet ILSVRC 2012 (I), and Pascal VOC 2012 (P). There are 50 images in each category and 600 images in each domain. We use all domain combinations and build 6 transfer tasks: I → P, P → I, I → C, C → I, C → P, and P → C. Different from the Office-31 where different domains are of different sizes, the three domains in this dataset are of the same size.

**ImageCLEF-DA是ImageCLEF 2014域适应挑战的基准，通过选择由三个公共数据集共享的12个常见类别进行组织，每个类别被视为一个域：Caltech-256（C），ImageNet ILSVRC 2012（I）和Pascal VOC 2012（P）。每个类别中有50个图像，每个域中有600个图像。我们使用所有域组合并构建6个传输任务：I→P，P→I，I→C，C→I，C→P和P→C。不同于Office-31，其中不同的域具有不同的大小，此数据集中的三个域具有相同的大小。**

Office-Home   This is a very challenging dataset for transfer learning evaluation, which consists of around 15,500 images in total from 65 categories of everyday objects in office and home settings coming from 4 significantly different domains: Artistic images (Ar), Clip Art (Cl), Product images (Pr) and Real-World images (Rw).

**Office-Home这是一个非常具有挑战性的转移学习评估数据集，它包含来自4个显着不同领域的办公室和家庭环境中65种日常物品的大约15,500张图像：艺术图像（Ar），剪贴画（Cl） ），产品图像（Pr）和真实世界图像（Rw）。**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_14.png)

We perform comparative study of our conditional domain adversarial network (CDAN) against shallow transfer learning and state-of-the-art deep transfer learning methods: Transfer Component Analysis (TCA) , Geodesic Flow Kernel (GFK), Deep Adaptation Network (DAN) , Residual Transfer Network (RTN) , Joint Adaptation Network (JAN) , Domain Adversarial Neural Network (DANN) , and Adversarial Discriminative Domain Adaptation (ADDA) .

**我们对浅层转移学习和最先进的深度转移学习方法进行了条件域对抗网络（CDAN）的比较研究：传递分量分析（TCA），测地线流核（GFK），深度适应网络（DAN） ，残留传输网络（RTN），联合适应网络（JAN），域对抗性神经网络（DANN）和对抗性判别域适应（ADDA）。**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_15.png)

TCA learns a transfer feature space by Kernel PCA with linear-MMD penalty. GFK interpolates across an infinite number of intermediate subspaces to bridge the source and target subspaces. For these shallow transfer methods, we use SVM as the base classifier. DAN learns transferable features by embedding the deep features of multiple domain-specific layers to reproducing kernel Hilbert spaces (RKHSs) and matching different distributions using multi-kernel MMD. JAN extends DAN by matching the joint distributions of the deep activations in multiple domain-specific layers using Joint MMD. RTN learns transferable features and adaptive classifiers jointly via deep residual learning . DANN enables domain adversarial learning by adapting a single layer of deep networks as in Eq. (3), which matches the source and target features by making them indistinguishable for a domain discriminator. ADDA is a generalized framework for adversarial domain adaptation, which combines discriminative modeling, untied weight sharing, and GAN loss for asymmetric adaptation.

**TCA通过具有线性MMD惩罚的内核PCA学习传输特征空间。 GFK在无限数量的中间子空间中进行插值，以桥接源子空间和目标子空间。对于这些浅层传输方法，我们使用SVM作为基础分类器。 DAN通过嵌入多个特定于域的层的深层特征来再现内核希尔伯特空间（RKHS）并使用多核MMD匹配不同的分布来学习可转移特征。 JAN通过使用Joint MMD匹配多个特定于域的层中的深度激活的联合分布来扩展DAN。 RTN通过深度残差学习共同学习可转移特征和自适应分类器。 DANN通过调整单层深层网络来实现域对抗性学习，如公式1所示。 （3），它通过使它们与域鉴别器无法区分来匹配源和目标特征。 ADDA是对抗域适应的通用框架，它结合了判别建模，无条件权重共享和非对称适应的GAN损失。**

We follow standard evaluation protocols for unsupervised domain adaptation . We use all labeled source examples and all unlabeled target examples for all datasets. We compare the average classification accuracy of each method on three random experiments, and report the standard error by all experiments of the same transfer task. We conduct importance-weighted cross-validation  for all baseline methods if their model selection strategies are not specified, and for our CDAN models to select parameter λ. As our CDAN models perform stably under different parameter configurations, we can fix λ = 1 throughout all experiments.
For MMD-based methods (TCA, DAN, RTN, and JAN), we use Gaussian kernel with bandwidth set to the median pairwise squared distances on the training data . We examine the influence of deep architectures for domain adaptation by exploring AlexNet  and ResNet-50  as base architectures. For shallow methods, we follow DeCAF  to use as deep representations outputs of layer fc7 in AlexNet and layer pool5 in ResNet.

**我们遵循无监督域适应的标准评估协议。我们对所有数据集使用所有标记的源示例和所有未标记的目标示例。我们比较了三种随机实验中每种方法的平均分类精度，并通过相同转移任务的所有实验报告标准误差。如果未指定模型选择策略，我们对所有基线方法进行重要性加权交叉验证，并为我们的CDAN模型选择参数λ。由于我们的CDAN模型在不同的参数配置下稳定运行，我们可以在所有实验中确定λ= 1。**
**对于基于MMD的方法（TCA，DAN，RTN和JAN），我们使用高斯核，其带宽设置为训练数据上的中间成对平方距离。我们通过探索AlexNet和ResNet-50作为基础架构来研究深层架构对域适应的影响。对于浅层方法，我们遵循DeCAF用作AlexNet中层fc7和ResNet中层池5的深层表示输出。**

We implement all deep methods based on Caffe for AlexNet and on PyTorch for ResNet. We finetune from AlexNet and ResNet models pre-trained on the ImageNet dataset . We fine-tune all convolutional layers and train the classifier layer through back propagation, where the classifier is trained from scratch with learning rate 10 times that of the other layers. We adopt mini-batch stochastic gradient descent (SGD) with momentum of 0.9 and the learning rate annealing strategy of DANN : the learning rate is adjusted during SGD by , where p is the training progress linearly changing from 0 to 1, and η0 = 0.01,α = 10, β = 0.75 are optimized by the importance-weighted cross-validation. We adopt progressive strategy for discriminator, that increases λ from 0 to 1 by multiplying λ with  .

**我们为AlexNet和PyTorch for ResNet实现了基于Caffe的所有深层方法。 我们通过在ImageNet数据集上预先训练的AlexNet和ResNet模型进行微调。 我们微调所有卷积层并通过反向传播训练分类器层，其中分类器从头开始训练，学习率是其他层的10倍。 我们采用动量为0.9的小批量随机梯度下降（SGD）和DANN的学习率退火策略：在SGD期间调整学习率，其中p是从0到1线性变化的训练进度，并且η0= 0.01 ，α= 10，β= 0.75通过重要性加权交叉验证进行优化。 我们采用渐进策略进行鉴别，通过将λ乘以λ将λ从0增加到1。**

### 结果

The classification accuracies on the Office-31 dataset based on AlexNet and ResNet are reported in Table 1. For fair comparison, results of baselines are directly reported from their original papers wherever available. The CDAN models significantly outperform all comparison methods on most transfer tasks, where CDAN-M is the top-performing variant and CDAN-RM performs slightly worse. It is desirable that CDAN promotes the classification accuracies substantially on hard transfer tasks, e.g. A → W, A → D, D → A, and W→ A, where the source and target are substantially different, and produce comparable performance on easy transfer tasks, D → W and W → D, where the source and target are similar, based on the dataset specification in .

**表1中报告了基于AlexNet和ResNet的Office-31数据集的分类准确度。为了公平比较，基线的结果直接从其原始论文中报告。 在大多数传输任务中，CDAN模型明显优于所有比较方法，其中CDAN-M是性能最佳的变体，而CDAN-RM的性能稍差。 期望CDAN基本上在硬传输任务上提高分类准确度，例如， A→W，A→D，D→A和W→A，其中源和目标基本不同，并且在易于传输任务D→W和W→D上产生相当的性能，其中源和目标相似 ，基于数据集规范。**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_16.png)

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_17.png)

The classification accuracies on the ImageCLEF-DA dataset are reported in Table 2. Our CDAN models outperform the comparison methods on most transfer tasks, but with smaller rooms of improvement than on the Office-31 dataset. This is reasonable since the three domains in the ImageCLEF-DA dataset are of equal size and balanced in each category, and are visually more similar than the Office-31 dataset. These characteristics make it much easier for domain adaptation.

**表2中报告了ImageCLEF-DA数据集的分类准确度。我们的CDAN模型在大多数传输任务上的表现优于比较方法，但改进的空间比Office-31数据集小。 这是合理的，因为ImageCLEF-DA数据集中的三个域在每个类别中具有相同的大小和平衡，并且在视觉上比Office-31数据集更相似。 这些特征使域适应变得更加容易。**

The classification accuracies on the Office-Home dataset are reported in Table 3. Our CDAN models substantially outperform the comparison methods on most transfer tasks, and with larger rooms of improvement than on the Office-31 dataset. An interpretation is that the four domains in the Office-Home dataset are with more categories, are visually very dissimilar with each other, and are difficult in each domain with much lower in-domain classification accuracy, as described in its original paper (Venkateswara et al., 2017). Since domain alignment is category agnostic in previous work, it is possible that the aligned domains are not classification friendly in the presence of large number of categories. It is very desirable that CDAN yields larger improvements on such difficult domain adaptation tasks, which highlights the power of domain adversarial adaptation conditioned on complex multimode structures such as classifier predictions.

**表3中报告了Office-Home数据集的分类准确度。我们的CDAN模型在大多数传输任务上的表现基本上优于比较方法，并且具有比Office-31数据集更大的改进空间。 一种解释是，Office-Home数据集中的四个域具有更多类别，在视觉上彼此非常不相似，并且在每个域中难以具有更低的域内分类准确性，如其原始论文中所述（Venkateswara et al。，2017）。 由于在先前的工作中域对齐与类别无关，因此在存在大量类别的情况下，对齐的域可能不是分类友好的。 非常期望CDAN在这种困难的域适应任务上产生更大的改进，这突出了以复杂的多模结构（例如分类器预测）为条件的域对抗自适应的能力。**

### 分析

Ablation Study We investigate different sampling of the random matrices in the randomized multilinear conditioning strategies. We testify CDAN-RM (bernoulli), CDAN-RM (gaussian), and CDAN-RM (uniform), with their random matrices sampled only once from Bernoulli, Gaussian, and Uniform distributions, respectively. Table 4 demonstrates that CDAN-RM (uniform) performs best across all variants, confirming theoretical approximation error in Theorem 1. We further investigate CDAN-M (focal loss) and CDAN-M (no focal loss). Table 4 demonstrates that CDAN-M (focal loss) outperforms CDAN-M (no focal loss) significantly, validating our efficacy of the inverse Focal Loss in Eq(10).

**消融研究我们研究随机多线性调节策略中随机矩阵的不同采样。 我们证明了CDAN-RM（bernoulli），CDAN-RM（高斯）和CDAN-RM（统一），它们的随机矩阵分别只从Bernoulli，Gaussian和Uniform分布中采样一次。 表4表明CDAN-RM（均匀）在所有变体中表现最佳，证实了定理1中的理论近似误差。我们进一步研究CDAN-M（焦点丢失）和CDAN-M（无焦点丢失）。 表4表明CDAN-M（焦点丢失）明显优于CDAN-M（无焦点丢失），证实了我们在方程（10）中反向焦点丢失的功效。**

Conditioning Strategies Besides our multilinear conditioning strategy, one may be curious on other conditioning strategies such as element-wise sum/product, concatenation. Note that element-wise operations are not applicable due to different dimensions in domain-specific feature layer f and classifier layer g. We investigate DANN-f and DANN-g with domain discriminator plugged in the feature layer f and classifier layer g, DANN-[f,g] with domain discriminator imposed on the concatenation of f and g. Figure 4(a) shows the accuracies on A → W and A → D based on ResNet-50.The concatenation strategy is not successful, since it cannot capture full interactions across feature representations and classifier predictions, which are key to domain adaptation.

**调节策略除了我们的多线性调节策略，人们可能会对其他调节策略感到好奇，例如元素总和/产品，连接。 注意，由于特定于域的特征层f和分类器层g中的不同维度，因此元素操作不适用。 我们研究了DANN-f和DANN-g，其中域鉴别器插入了特征层f和分类器层g，DANN- [f，g]，并且在f和g的串联上施加了域鉴别器。 图4（a）显示了基于ResNet-50的A→W和A→D的精度。连接策略不成功，因为它无法捕获特征表示和分类器预测之间的完全交互，这是域适应的关键。**

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_18.jpg)

![](https://raw.githubusercontent.com/CLAY2333/blog_md_img/master/CADA_19.jpg)

Distribution Discrepancy The domain adaptation theory suggests A-distance as a measure of distribution discrepancy . The proxy A-distance is defined as , where  is the generalization error of a classifier trained to discriminate source and target. Figure 4(b) shows dA on tasks A → W, W → D with features of ResNet, DANN, and CDAN-M. We observe that dA on CDAN-M features is much smaller than dA on ResNet and DANN features, implying that CDAN-M features can reduce the domain gap more effectively. As domains W and D are similar, dA of task W → D is smaller than that of A → W, which explains the accuracy difference.

**分布差异领域适应理论建议将A距离作为分布差异的度量。 代理A-距离被定义为，其中是被训练以区分源和目标的分类器的泛化误差。 图4（b）显示了具有ResNet，DANN和CDAN-M特征的任务A→W，W→D的dA。 我们观察到CDAN-M特征上的dA远小于ResNet和DANN特征上的dA，这意味着CDAN-M特征可以更有效地减少域间隙。 由于域W和D相似，任务W→D的dA小于A→W的dA，这解释了精度差异。**

Convergence Performance We testify the convergence performance of ResNet, DANN, and CDAN. Figure 4(c) shows test errors of different methods on task A → W. We observe that CDAN enjoys faster convergence than DANN, while CDAN-M converges faster than CDAN-RM. Note that CDAN-M deals with high-dimensional multilinear map, thus each iteration costs slightly more time than CDAN-RM, while CDAN-RM has similar computational cost as DANN.

**融合性能我们证明了ResNet，DANN和CDAN的融合性能。图4（c）显示了任务A→W上不同方法的测试误差。我们观察到CDAN比DANN更快收敛，而CDAN-M收敛速度比CDAN-RM快。请注意，CDAN-M处理高维多线性映射，因此每次迭代的成本略高于CDAN-RM，而CDAN-RM的计算成本与DANN相似。**

Feature Visualization We visualize by t-SNE in Figures 5(a)–5(d) the network activations of task A → W (31 classes) by ResNet, DANN, CDAN-f, and CDAN-fg. By ResNet features, the source and target are not well aligned. By DANN features, the source and target are better aligned, but different categories are not well discriminated. By CDAN-f features, the source and target are better aligned while different categories are better discriminated. By CDAN-fg features, the source and target are perfectly aligned while different categories are perfectly discriminated. This evidence shows the benefit of conditioning domain adversarial adaptation on discriminative predictions.

**特征可视化我们通过图5（a）-5（d）中的t-SNE可视化ResNet，DANN，CDAN-f和CDAN-fg的任务A→W（31类）的网络激活。通过ResNet功能，源和目标没有很好地对齐。通过DANN特征，源和目标更好地对齐，但不同的类别没有很好地区分。通过CDAN-f特征，源和目标更好地对齐，而不同类别被更好地区分。通过CDAN-fg特征，源和目标完全对齐，而不同的类别被完全区分。该证据表明调节域对抗性适应对判别性预测的益处。**

### 结论

This paper presented conditional domain adversarial network (CDAN), a novel approach to conditional adversarial domain adaptation. Unlike previous adversarial adaptation methods that solely match the feature representation across domains and may be trapped by the mode collapse difficulty, the proposed approach further conditions the adversarial domain adaptation on discriminative information to enable finer-grained alignment of multimode structures. Extensive experiments testified the efficacy of the proposed approach.

**本文提出了条件域对抗网络（CDAN），一种有条件的对抗域适应的新方法。 不同于以前的对抗性自适应方法，其仅跨越域的特征表示并且可能被模式崩溃困难所困，所提出的方法进一步调节对抗性信息上的对抗域适应以实现多模式结构的更细粒度对齐。 大量实验证明了所提方法的有效性。**