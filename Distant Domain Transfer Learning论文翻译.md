---
title: Distant Domain Transfer Learning论文翻译
date: 2018-11-05 14:42:39
categories: 论文翻译
tags: [机器学习,深度学习,迁移学习,论文翻译]
---

<Excerpt in index | 首页摘要> 

这次翻译的是2017年在AAAI上发表的关于大跨度的迁移学习的论文，本文的角度是从人脸到飞机的识别。翻译仅仅代表个人的，如果有误请指出。

<!-- more -->

<The rest of contents | 余下全文>

# Distant Domain Transfer Learning

## 摘要

In this paper, we study a novel transfer learning problem termed Distant Domain Transfer Learning (DDTL). Different from existing transfer learning problems which assume that there is a close relation between the source domain and the target domain, in the DDTL problem, the target domain can be totally different from the source domain. For example, the source domain classifies face images but the target domain distinguishes plane images. Inspired by the cognitive process of human where two seemingly unrelated concepts can be connected by learning intermediate concepts gradually, we propose a Selective Learning Algorithm (SLA) to solve the DDTL problem with supervised autoencoder or supervised convolutional autoencoder as a base model for handling different types of inputs. Intuitively, the SLA algorithm selects usefully unlabeled data gradually from intermediate domains as a bridge to break the large distribution gap for transferring knowledge between two distant domains. Empirical studies on image classification problems demonstrate the effectiveness of the proposed algorithm, and on some tasks the improvement in terms of the classification accuracy is up to 17% over “non-transfer” methods.

在本文中，我们研究了一种新的转移学习问题，称为远程域转移学习（DDTL）。与现有的转移学习问题不同，假设源域和目标域之间存在密切关系，在DDTL问题中，目标域可能与源域完全不同。例如，源域对面部图像进行分类，但目标域区分平面图像。受人类认知过程的启发，通过逐渐学习中间概念可以连接两个看似无关的概念，我们提出了一种选择性学习算法（SLA），用监督自动编码器或监督卷积自动编码器作为处理不同类型的基础模型来解决DDTL问题。投入直观地，SLA算法从中间域逐渐选择有用的未标记数据作为桥梁，以打破用于在两个远域之间传递知识的大分布间隙。对图像分类问题的实证研究证明了所提算法的有效性，并且在某些任务上，分类精度方面的改进比“非转移”方法高达17％。

## 简介

Introduction Transfer Learning, which borrows knowledge from a source domain to enhance the learning ability in a target domain, has received much attention recently and has been demonstrated to be effective in many applications. An essential requirement for successful knowledge transfer is that the source domain and the target domain should be closely related. This relation can be in the form of related instances, features or models, and measured by the KL-divergence or A-distnace (Blitzer et al. 2008). For two distant domains where no direct relation can be found, transferring knowledge between them forcibly will not work. In the worst case, it could lead to even worse performance than ‘non-transfer’ algorithms in the target domain, which is the ‘negative transfer’ phenomena (Rosenstein et al. 2005; Pan and Yang 2010). For example, online photo sharing communities, such as Flickr and Qzone1 , generate vast amount of images as well as their tags. However, due to the diverse

转移学习从源域借用知识以提高目标域的学习能力，最近受到了很多关注，并且已被证明在许多应用中都是有效的。 成功的知识转移的基本要求是源域和目标域应该密切相关。 这种关系可以是相关实例，特征或模型的形式，并通过KL-分歧或A-distnace来测量（Blitzer等人，2008）。 对于无法找到直接关系的两个遥远的领域，强行转移它们之间的知识是行不通的。 在最坏的情况下，它可能导致比目标域中的“非转移”算法更糟糕的性能，这是“负转移”现象（Rosenstein等人2005; Pan and Yang 2010）。 例如，在线照片共享社区（如Flickr和Qzone1）会生成大量图像以及标签。 但是，由于多样化

![ex2-2](http://www.lzyclay.cn/md_img/论文_人脸迁移到飞机1.jpg)

Figure 1: The tag distribution of the uploaded images at Qzone. In the first task, we transfer knowledge between cat and tiger images. A transfer learning algorithm achieves much better performance than some supervised learning algorithm. In the second task, we transfer knowledge between face and airplane images. The transfer learning algorithm fails as it achieves worse performance than the supervised learning algorithm. When applying our proposed SLA algorithm, however, we find that our model achieves much better performance.

图1：Qzone上传图像的标签分布。 在第一项任务中，我们在猫和老虎图像之间传递知识。 传递学习算法比一些监督学习算法实现了更好的性能。 在第二项任务中，我们在面部和飞机图像之间传递知识。 传输学习算法失败，因为它实现了比监督学习算法更差的性能。 然而，当应用我们提出的SLA算法时，我们发现我们的模型实现了更好的性能。

interests of users, the tag distribution is often long-tailed, which can be verified by our analysis in Figure 1 on the tag distribution of the uploaded images at Qzone from January to April in 2016. For the tags in the head part, we can build accurate learners as there are plenty of labeled data but in the tail part, due to the scarce labeled data, the learner for each tag usually has no satisfactory performance. In this case, we can adopt transfer learning algorithms to build accurate classifiers for tags in the tail part by reusing knowledge in the head part. When the tag in the tail part is related to that in the head part, this strategy usually works very well. For example, as shown in Figure 1, we can build an accurate tiger classifier by transferring knowledge from cat images when we have few labeled tiger images, where the performance improvement is as large as 24% compared to some supervised learning algorithm learned from labeled tiger images only. However, if the two tags (e.g., face and airplane images) are totally unrelated from our perspective, existing transfer learning algorithms such as (Patel et al. 2015) fail as shown in Figure 1. One reason for the failure of existing transfer learning algorithms is that the two domains, face and airplane, do not share any common characteristic in shape or other aspects, and hence they are conceptually distant, which violates the assumption of existing transfer learning works that the source domain and the target domain are closely related

对于用户的兴趣，标签分发通常是长尾的，这可以通过我们在图1中对2016年1月到4月Qzone上传图像的标签分布的分析进行验证。对于头部的标签，我们可以建立准确的学习者，因为有大量的标签数据，但在尾部，由于标签数据稀少，每个标签的学习者通常没有令人满意的表现。在这种情况下，我们可以采用转移学习算法，通过重用头部的知识，为尾部的标签建立精确的分类器。当尾部的标签与头部的标签相关时，这种策略通常效果很好。例如，如图1所示，我们可以通过从猫图像转移知识来构建准确的老虎分类器，当我们标记虎图像很少时，与从标记老虎学到的一些监督学习算法相比，性能提高高达24％仅限图像。但是，如果两个标签（例如，面部和飞机图像）与我们的观点完全不相关，那么现有的转移学习算法（如Patel等2015）就会失败，如图1所示。现有转移学习失败的原因之一算法是两个领域，即面部和飞机，在形状或其他方面不共享任何共同特征，因此它们在概念上是遥远的，这违反了现有转移学习工作的假设，源域和目标域密切相关

In this paper, we focus on transferring knowledge between two distant domains, which is referred to Distant Domain Transfer Learning (DDTL). The DDTL problem is critical as solving it can largely expand the application scope of transfer learning and help reuse as much previous knowledge as possible. Nonetheless, this is a difficult problem as the distribution gap between the source domain and the target do- main is large. The motivation behind our solution to solve the DDTL problem is inspired by human’s ‘transitivity’ learning and inference ability (Bryant and Trabasso 1971). That is, people transfer knowledge between two seemingly unrelated concepts via one or more intermediate concepts as a bridge. 

在本文中，我们专注于在两个远程域之间传递知识，这被称为远程域转移学习（DDTL）。 DDTL问题至关重要，因为解决它可以在很大程度上扩展转移学习的应用范围，并帮助尽可能多地重用先前的知识。尽管如此，这是一个难题，因为源域和目标域之间的分布差距很大。我们解决DDTL问题的解决方案背后的动机是受到人类“传递性”学习和推理能力的启发（Bryant和Trabasso，1971）。也就是说，人们通过一个或多个中间概念作为桥梁，在两个看似无关的概念之间传递知识。

Along this line, there are several works aiming to solve the DDTL problem. For instance, Tan et al. (2015) introduce annotated images to bridge the knowledge transfer between text data in the source domain and image data in the target domain, and Xie et al. (2016) predict the poverty based on the daytime satellite imagery by transferring knowledge from an object classification task with the help of some nighttime light intensity information as an intermediate bridge. Those studies assume that there is only one intermediate domain and that all the data in the intermediate domain are helpful. However, in some cases the distant domains can only be re- lated via multiple intermediate domains. Exploiting only one intermediate domain is not enough to help transfer knowl- edge across long-distant domains. Moreover, given multiple intermediate domains, it is highly possible that only a sub- set of data from each intermediate domain is useful for the target domain, and hence we need an automatic selection mechanism to determine the subsets. 

沿着这条线，有几个旨在解决DDTL问题的工作。例如，Tan等人。 （2015）引入注释图像来桥接源域中的文本数据和目标域中的图像数据之间的知识转移，并且Xie等人。 （2016）通过在一些夜间光强度信息作为中间桥梁的帮助下从对象分类任务转移知识，基于白天卫星图像预测贫困。这些研究假设只有一个中间域，并且中间域中的所有数据都是有用的。但是，在某些情况下，远程域只能通过多个中间域进行关联。仅利用一个中间域不足以帮助跨越远程域转移知识。此外，给定多个中间域，很可能只有来自每个中间域的数据子集对目标域有用，因此我们需要一种自动选择机制来确定子集。

In this paper, to solve the DDTL problem in a better way, we aim to transfer knowledge between distant domains by gradually selecting multiple subsets of instances from a mix- ture of intermediate domains as a bridge. We use the recon- struction error as a measure of distance between two domains. That is, if the data reconstruction error on some data points in the source domain is small based on a model trained on the target domain, then we consider that these data points in the source domain are helpful for the target domain. Based on this measure, we propose a Selective Learning Algorithm (SLA) for the DDTL problem, which simultaneously selects useful instances from the source and intermediate domains, learns high-level representations for selected data, and trains a classifier for the target domain. The learning process of SLA is an iterative procedure that selectively adds new data points from intermediate domains and removes unhelpful data in the source domain to revise the source-specific model changing towards a target-specific model step by step until some stopping criterion is satisfied.

在本文中，为了更好地解决DDTL问题，我们的目标是通过逐步从中间域混合中选择多个实例子集作为桥梁，在远程域之间传递知识。我们使用重构误差作为两个域之间距离的度量。也就是说，如果源域中某些数据点上的数据重建错误很小，则基于在目标域上训练的模型，那么我们认为源域中的这些数据点对目标域有用。基于此度量，我们为DDTL问题提出了一种选择性学习算法（SLA），它同时从源域和中间域中选择有用的实例，学习所选数据的高级表示，并训练目标域的分类器。 SLA的学习过程是一个迭代过程，它选择性地从中间域添加新数据点，并删除源域中的无用数据，以修改特定于源的模型逐步向目标特定模型转变，直到满足一些停止标准。

The contributions of this paper are three-fold. Firstly, to our best knowledge, this is the first work that studies the DDTL problem by using a mixture of intermediate domains. Secondly, we propose an SLA algorithm for DDTL. Third- ly, we conduct extensive experiments on several real-world datasets to demonstrate the effectiveness of the proposed algorithm. 

 本文的贡献有三个方面的。 首先，据我们所知，这是第一项使用中间域混合物研究DDTL问题的工作。 其次，我们提出了一种用于DDTL的SLA算法。 第三，我们对几个真实数据集进行了大量实验，以证明所提算法的有效性。

## 相关工作

Typical transfer learning algorithms include instance weight- ing approaches (Dai et al. 2007) which select relevant data from the source domain to help the learning in the target domain, feature mapping approaches (Pan et al. 2011) which transform the data in both source and target domains into a common feature space where data from the two domains fol- low similar distributions, and model adaptation approach (Ay- tar and Zisserman 2011) which adapt the model trained in the source domain to the target domain. However, these ap- proaches cannot handle the DDTL problem as they assume that the source domain and the target domain are conceptually close. Recent studies (Yosinski et al. 2014; Oquab et al. 2014; Long et al. 2015) reveal that deep neural networks can learn transferable features for a target domain from a source do- main but they still assume that the target domain is closely related to the source domain. 

典型的转移学习算法包括实例加权方法（Dai et al.2007），它从源域中选择相关数据以帮助在目标域中学习，特征映射方法（Pan et al.2011）转换两者中的数据源和目标域进入一个共同的特征空间，其中来自两个域的数据遵循相似的分布，以及模型自适应方法（Ay-tar和Zisserman 2011），它们将源域中训练的模型适应目标域。但是，这些方法无法处理DDTL问题，因为它们假设源域和目标域在概念上是接近的。最近的研究（Yosinski等人2014; Oquab等人2014; Long等人2015）揭示了深度神经网络可以从源头学习目标域的可转移特征，但他们仍然认为目标域是紧密的与源域相关

The transitive transfer learning (TTL) (Tan et al. 2015; Xie et al. 2016) also learns from the target domain with the help of a source domain and an intermediate domain. In TTL, there is only one intermediate domain, which is selected by users manually, and all intermediate domain data are used. Different from TTL, our work automatically selects subsets from a mixture of multiple intermediate domains as a bridge across the source domain and the target domain. Transfer Learning with Multiple Source Domains (TLMS) (Mansour, Mohri, and Rostamizadeh 2009; Tan et al. 2013) leverages multiple source domains to help learning in the target domain, and aims to combine knowledge simultaneously transferred from all the source domains. The different between TLMS and our work is two-fold. First, all the source domains in TLMS have sufficient labeled data. Second, all the source domains in TLMS are close to the target domain. 

传递转移学习（TTL）（Tan等人2015; Xie等人2016）也在源域和中间域的帮助下从目标域学习。在TTL中，只有一个中间域，由用户手动选择，并使用所有中间域数据。与TTL不同，我们的工作自动从多个中间域的混合中选择子集作为跨源域和目标域的桥。多源域转移学习（TLMS）（Mansour，Mohri和Rostamizadeh 2009; Tan等人2013）利用多个源域来帮助在目标域中学习，并旨在结合从所有源域同时传输的知识。 TLMS与我们的工作之间的差异是双重的。首先，TLMS中的所有源域都有足够的标记数据。其次，TLMS中的所有源域都接近目标域。 

Semi-supervised autoencoder (SSA) (Weston, Ratle, and Collobert 2008; Socher et al. 2011) also aims to minimize both the reconstruction error and the training loss while learn- ing a feature representation. However, our work is different from SSA in three-fold. First, in SSA, both unlabeled and labeled data are from the same domain, while in our work, labeled data are from either the source domain or the target domain and unlabeled data are from a mixture of interme- diate domains, whose distributions can be very different to each other. Second, SSA uses all the labeled and unlabeled data for learning, while our work selectively chooses some unlabeled data from the intermediate domains and removes some labeled data from the source domain for assisting the learning in the target domain. Third, SSA does not have con- volutional layer(s), while our work uses convolutional filters if the input is a matrix or tensor. 

半监督自动编码器（SSA）（Weston，Ratle和Collobert 2008; Socher等人2011）也旨在在学习特征表示的同时最小化重建误差和训练损失。 但是，我们的工作与SSA有三个不同。 首先，在SSA中，未标记和标记的数据都来自同一个域，而在我们的工作中，标记的数据来自源域或目标域，未标记的数据来自中间域的混合，其分布可以是 彼此非常不同。 其次，SSA使用所有标记和未标记的数据进行学习，而我们的工作选择性地从中间域中选择一些未标记的数据，并从源域中删除一些标记数据，以帮助在目标域中进行学习。 第三，SSA没有卷积层，而我们的工作使用卷积滤波器，如果输入是矩阵或张量。

## 问题定义

We denote by S ={(x1,y1),··· ,(xnS,ynS)} the source SSSS domain labeled data of size nS , which is assumed to be suf- ficient enough to train an accurate classifier for the source domain, and by T = {(x1 ,y1 ),··· ,(xnT ,ynT )} the tar- TTTT get domain labeled data of size nT , which is assumed to be too insufficient to learn an accurate classifier for the target domain. Moreover, we denote by I = {x1, · · · , xnI } the II mixture of unlabeled data of multiple intermediate domains, where nI is assumed to be large enough. In this work, a domain corresponds to a concept or class for a specific clas- sification problem, such as face or airplane recognition from images. Without loss of generality, we suppose the classifica- tion problems in the source domain and the target domain are both binary. All data points are supposed to lie in the same feature space. Let pS (x), pS (y|x), pS (x, y) be the marginal, conditional and joint distributions of the source domain data, respectively, pT (x), pT (y|x), pT (x, y) be the parallel def- initions for the target domain, and pI (x) be the marginal distribution for the intermediate domains. In a DDTL prob- lem, we have

 我们用S = {（x1，y1），...，（xnS，ynS）}表示源SSSS域标记的大小为nS的数据，假设它足以训练源域的精确分类器，并且通过T = {（x1，y1），...，（xnT，ynT）}，tar-TTTT获得大小为nT的域标记数据，假设其不足以学习目标域的准确分类器。此外，我们用I = {x1，···，xnI}表示多个中间域的未标记数据的II混合物，其中假设nI足够大。在这项工作中，域对应于特定分类问题的概念或类，例如图像中的面部或飞机识别。在不失一般性的情况下，我们假设源域和目标域中的分类问题都是二元的。所有数据点应该位于相同的特征空间中。令pS（x），pS（y | x），pS（x，y）分别为源域数据的边缘，条件和联合分布，pT（x），pT（y | x），pT（x） ，y）是目标域的平行定义，pI（x）是中间域的边际分布。在DDTL问题中，我们有

pT (x) ̸= pS (x), pT (x) ̸= pI (x), and pT (y|x) ̸= pS (y|x). The goal of DDTL is to exploit the unlabeled data in the intermediate domains to build a bridge between the source and target domains, which are originally distant to each oth- er, and train an accurate classifier for the target domain by transferring supervised knowledge from the source domain with the help of the bridge. Note that not all the data in the intermediate domains are supposed to be similar to the source domain data, and some of them may be quite different. There- fore, simply using all the intermediate data to build the bridge may fail to work. 

pT（x）̸= pS（x），pT（x）̸= pI（x），pT（y | x）̸= pS（y | x）。 DDTL的目标是利用中间域中的未标记数据在源域和目标域之间建立桥梁，这些域最初远离其他域，并通过从目标域传输监督知识来训练目标域的精确分类器。 在桥的帮助下的源域。 请注意，并非中间域中的所有数据都应与源域数据类似，并且其中一些数据可能完全不同。 因此，仅使用所有中间数据来构建桥可能无法工作。

## 选择性学习算法(SLA)

在本节中，我们提出了提议的SLA。

### 自动编码器及其变体

As a basis component in our proposed method to solve the DDTL problem is the autoencoder (Bengio 2009) and its vari- ant, we first review them. An autoencoder is an unsupervised feed-forward neural network with an input layer, one or more hidden layers, and an output layer. It usually includes two processes: encoding and decoding. Given an input x ∈ Rq, an autoencoder first encodes it through an encoding function fe(·) to map it to a hidden representation, and then decodes it through a decoding function fd(·) to reconstruct x. The process of the autoencoder can be summarized as

作为我们提出的解决DDTL问题的方法的基础组件是autoencoder（Bengio 2009）及其变量，我们首先回顾它们。 自动编码器是无监督的前馈神经网络，具有输入层，一个或多个隐藏层和输出层。 它通常包括两个过程：编码和解码。 给定输入x∈Rq，自动编码器首先通过编码函数fe（·）对其进行编码以将其映射到隐藏表示，然后通过解码函数fd（·）对其进行解码以重建x。 自动编码器的过程可以概括为

encoding : h = fe(x), and decoding : xˆ = fd(h),

where xˆ is the reconstructed input to approximate x. The learning of the pair of encoding and decoding functions, fe(·) and fd(·), is done by minimizing the reconstruction error over all training data, i.e., 

其中x是近似x的重建输入。 通过最小化所有训练数据上的重建误差来完成对编码和解码功能对fe（·）和fd（·）的学习，即 ![](http://www.lzyclay.cn/md_img/论文_人脸迁移到飞机4.PNG)

After the pair of encoding and decoding functions are learned, the output of encoding function of an input x, i.e., h = fe(x), is considered as a higher-level and robust repre- sentation for x. Note that an autoencoder takes a vector as the input. When an input instance represented by a matrix or tensor, such as images, is presented to an autoencoder, the spatial information of the instance may be discarded. In this case, a convolutional autoencoder is more desired, and it is a variant of the autoencoder by adding one or more convolution- al layers to capture inputs, and one or more correspondingly deconvolutional layers to generate outputs. 

在学习了这对编码和解码函数之后，输入x的编码函数的输出，即h = fe（x），被认为是x的更高级和稳健的表示。 请注意，自动编码器将矢量作为输入。 当由诸如图像的矩阵或张量表示的输入实例被呈现给自动编码器时，可以丢弃该实例的空间信息。 在这种情况下，更期望卷积自动编码器，并且它是自动编码器的变体，其通过添加一个或多个卷积层来捕获输入，以及一个或多个相应的去卷积层以生成输出。

### Instance Selection via Reconstruction Error 通过重建误差选择实例

A motivation behind our proposed method is that in an ideal case, if the data from the source domain are similar and useful for the target domain, then one should be able to find a pair of encoding and decoding functions such that the reconstruction errors on the source domain data and the target domain data are both small. In practice, as the source domain and the target domain are distant, there may be only a subset of the source domain data is useful for the target domain. The situation is similar in the intermediate domains. Therefore, to select useful instances from the intermediate domains, and remove irrelevant instances from the source domain for the target domain, we propose to learn a pair of encoding and decoding functions by minimizing reconstruction errors on the selected instances in the source and intermediate domains, and all the instances in the target domain simultaneously. The objective function to be minimized is formulated as follows: 

我们提出的方法背后的动机是，在理想的情况下，如果来自源域的数据对于目标域是相似且有用的，那么应该能够找到一对编码和解码函数，使得重建错误在源域数据和目标域数据都很小。实际上，由于源域和目标域很远，因此可能只有源域数据的子集对目标域有用。中间域的情况类似。因此，要从中间域中选择有用的实例，并从源域中删除目标域的不相关实例，我们建议通过最小化源域和中间域中所选实例的重建错误来学习一对编码和解码函数，和目标域中的所有实例同时进行。要最小化的目标函数表述如下：

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机5.JPG)

where xˆiS , xˆiI and xˆiT are reconstructions of xiS , xiI and xi based on the auto encoder,vS =(v1,···,vnS)⊤,vI = (v1, · · · , vnI )⊤, and vi , vj ∈ {0, 1} are selection indicators for the i-th instance in the source domain and the j-th instance in the intermediate domains, respectively. When the value is 1, the corresponding instance is selected and otherwise unselected. The last term in the objective, R(vS , vT ), is a regularization term on vS and vT to avoid a trivial solution by setting all values of vS and vT to be zero. In this paper, we define R(vS , vT ) as

其中xiS，xiI和xiT是基于自动编码器的xiS，xiI和xi的重建，vS =（v1，...，vnS）⊤，vI =（v1，...，vnI）⊤和vi，vj ∈{0,1}分别是源域中的第i个实例和中间域中的第j个实例的选择指示符。 当值为1时，将选择相应的实例，否则将取消选择。 目标中的最后一项R（vS，vT）是vS和vT上的正则化项，通过将vS和vT的所有值设置为零来避免平凡解。 在本文中，我们将R（vS，vT）定义为

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机6.JPG)

Minimizing this term is equivalent to encouraging to select many instances as possible from the source and intermediate domains. Two regularization parameters, λS and λI , control the importance of this regularization term. Note that more useful instances are selected, more robust hidden representa- tions can be learned through the autoencoder.

最小化此术语等同于鼓励从源域和中间域中选择尽可能多的实例。 两个正则化参数λS和λI控制该正则化项的重要性。 请注意，选择了更多有用的实例，可以通过自动编码器学习更强大的隐藏表示。

### 结合边信息

By solving the minimization problem (1), one can select use- ful instances from the source and intermediate domains for the target domain through vS and vT , and learn high-level hidden representations for data in different domains through the encoding function, i.e., h=fe(x), simultaneously. How- ever, the learning process is in an unsupervised manner. As a result, the learned hidden representations may not be rele- vant to the classification problem in the target domain. This motivates us to incorporate side information into the learn- ing of the hidden representations for different domains. For the source and target domains, labeled data can be used as the side information, while for the intermediate domains, there is no label information. In this work, we consider the predictions on the intermediate domains as the side informa- tion, and use the confidence on the predictions to guide the learning of the hidden representations. To be specific, we propose to incorporate the side information into learning by minimizing the following function:

通过求解最小化问题（1），可以通过vS和vT从源域和中间域中为目标域选择有用的实例，并通过编码函数学习不同域中数据的高级隐藏表示，即h = fe（x），同时。然而，学习过程是以无人监督的方式进行的。因此，学习的隐藏表示可能与目标域中的分类问题无关。这促使我们将辅助信息结合到学习不同领域的隐藏表示中。对于源域和目标域，标记数据可以用作辅助信息，而对于中间域，没有标签信息。在这项工作中，我们将中间域的预测视为侧面信息，并使用对预测的信心来指导隐藏表示的学习。具体而言，我们建议通过最小化以下功能将辅助信息纳入学习：

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机3.jpg)



其中fc（·）是输出分类概率的分类函数，g（·）是熵函数，定义为g（z）= -z ln z - （1 - z）ln（1 - z）0≤ z≤1，用于选择中间域中高预测置信度的实例。

### 总体目标功能

By combining the two objectives in Eqs. (1) and (2), we obtain the final objective function for DDTL as follows:

通过将方程式中的两个目标结合起来。 （1）和（2），我们获得DDTL的最终目标函数如下：

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机7.JPG)

where v = {vS , vT }, and Θ denotes all parameters of the functions fc(·), fe(·), and fd(·).To solve problem (3), we use the block coordinate dece- dent (BCD) method, where in each iteration, variables in each block are optimized sequentially while keeping other variables fixed. In problem (3), there are two blocks of vari- ables: Θ and v. When the variables in v are fixed, we can update Θ using the Back Propagation (BP) algorithm where the gradients can be computed easily. Alternatingly, when the variables in Θ are fixed, we can obtain an analytical solution for v as follows,

其中v = {vS，vT}，Θ表示函数fc（·），fe（·）和fd（·）的所有参数。为了解决问题（3），我们使用块坐标值（BCD） ）方法，其中在每次迭代中，每个块中的变量被顺序优化，同时保持其他变量固定。 在问题（3）中，存在两个变量块：Θ和v。当v中的变量固定时，我们可以使用反向传播（BP）算法更新Θ，其中可以容易地计算梯度。 或者，当Θ中的变量固定时，我们可以得到v的解析解，如下所示，

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机8.JPG)

Based on Eq. (4), we can see that for the data in the source domain, only those with low reconstruction errors and low training losses will be selected during the optimization pro- cedure. Similarly, based on Eq. (5), it can be found that for the data in the intermediate domains, only those with low reconstruction errors and high prediction confidence will be selected.

An intuitive explanation of this learning strategy is two- fold: 1) When updating v with a fixed Θ, “useless” data in the source domain will be removed, and those intermediate data that can bridge the source and target domains will be selected for training; 2) When updating Θ with fixed v, the model is trained only on the selected “useful” data samples.The overall algorithm for solving problem (3) is summarized in Algorithm 1.

基于Eq。 （4），我们可以看到，对于源域中的数据，在优化过程中只会选择那些重建误差低，训练损耗低的人。 同样，基于Eq。（5），可以发现，对于中间域中的数据，仅选择具有低重建误差和高预测置信度的数据。

这种学习策略的直观解释有两个：1）当用固定的Θ更新v时，将删除源域中的“无用”数据，并且将选择可以桥接源域和目标域的那些中间数据训练; 2）当用固定v更新Θ时，仅对所选择的“有用”数据样本训练模型。在算法1中总结了用于解决问题（3）的总体算法。

The deep learning architecture corresponding to problem (3) is illustrated in Figure 2. From Figure 2, we note that except for the instance selection component v, the rest of the architecture in Figure 2 can be viewed as a generalization of an autoencoder or a convolutional autoencoder by incorpo- rating the side information, respectively. In the sequel, we refer to the architecture (except for the instances selection component) using an antoencoder for fe(·) and fd(·) as SAN (Supervised AutoeNcoder), and the one using a convolutional antoencoder as SCAN (Supervised Convolutional AutoeN- coder).

与问题（3）相对应的深度学习架构如图2所示。从图2中我们注意到，除了实例选择组件v之外，图2中的其余架构可以被视为自动编码器的概括或者 卷积自动编码器分别包含辅助信息。 在后续内容中，我们使用针对fe（·）和fd（·）的antencoder作为SAN（监督AutoeNcoder），使用卷积自动编码器作为SCAN（监督卷积AutoeN）的体系结构（实例选择组件除外） - 编码员）。

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机9.JPG)

## 实验

In this section, we conduct empirical studies to evaluate the proposed SLA algorithm from three aspects.2 Firstly, we test the effectiveness of the algorithm when the source domain and the target domain are distant. Secondly, we visualize some selected intermediate data to understand how the al- gorithm works. Thirdly, we evaluate the importance of the learning order of intermediate instances by comparing the order generated by SLA with other manually designed orders.

在本节中，我们进行了实证研究，从三个方面评估了所提出的SLA算法.2首先，我们测试了源域和目标域距离较远的算法的有效性。 其次，我们可视化一些选定的中间数据，以了解算法的工作原理。 第三，我们通过比较SLA生成的订单与其他手动设计的订单来评估中间实例的学习顺序的重要性。

### 基线方法

Three categories of methods are used as baselines. 

**Supervised Learning **In this category, we choose two su- pervised learning algorithms, SVM and convolutional neural networks (CNN) (Krizhevsky, Sutskever, and Hinton 2012), as baselines. For SVM, we use the linear kernel. For CNN,we implement a network that is composed of two convolu-tional layers with kernel size 3 × 3, where each convolutional layer is followed by a max pooling layer with kernel size 2 × 2, a fully connected layer, and a logistic regression layer.Transfer Learning In this category, we choose five transfer learning algorithms including Adaptive SVM  (ASVM)(Aytar and Zisserman 2011), Geodesic Flow Kernel (GFK) (Gong et al. 2012), Landmark (LAN) (Gong, Grauman, and Sha 2013), Deep Transfer Learning (DTL) (Yosinski et al. 2014), and Transitive Transfer Learning (TTL) (Tan et al. 2015). For ASVM, we use all the source domain and target domain la- beled data to train a target model. For GFK, we first generate a series of intermediate spaces using all the source domain and target domain data to learn a new feature space, and then train a model in the space using all the source domain and target domain labeled data. For LAN, we first select a subset of labeled data from the source domain, which has a similar distribution as the target domain, and then use all the selected data to facilitate knowledge transfer. For DTL, we first train a deep model in the source domain, and then train another deep model for the target domain by reusing the first several layers of the source model. For TTL, we use all the source domain, target domain and intermediate domains data to learn a model.

三类方法用作基线。

监督学习在这一类中，我们选择两种监督学习算法，SVM和卷积神经网络（CNN）（Krizhevsky，Sutskever和Hinton 2012）作为基线。对于SVM，我们使用线性内核。对于CNN，我们实现了一个由两个卷积为3×3的卷积层组成的网络，其中每个卷积层后面跟着一个内核大小为2×2的最大池层，一个完全连接的层和一个逻辑回归layer.Transfer Learning在这个类别中，我们选择五种转移学习算法，包括自适应SVM（ASVM）（Aytar和Zisserman 2011），Geodesic Flow Kernel（GFK）（Gong et al.2012），Landmark（LAN）（Gong，Grauman，和Sha 2013），深度转移学习（DTL）（Yosinski等人2014）和传递转移学习（TTL）（Tan等人2015）。对于ASVM，我们使用所有源域和目标域标签数据来训练目标模型。对于GFK，我们首先使用所有源域和目标域数据生成一系列中间空间以学习新的特征空间，然后使用所有源域和目标域标记的数据在空间中训练模型。对于LAN，我们首先从源域中选择标记数据的子集，其具有与目标域类似的分布，然后使用所有选择的数据来促进知识传输。对于DTL，我们首先在源域中训练深度模型，然后通过重用源模型的前几个层来训练目标域的另一个深度模型。对于TTL，我们使用所有源域，目标域和中间域数据来学习模型。

**Self-taught Learning (STL)** We apply the autoencoder or convolutional autoencoder on all the intermediate domains data to learn a universal feature representation, and then train a classifier with all the source domain and target domain labeled data with the new feature representation.
Among the baselines, CNN can receive tensors as inputs, DTL and STL can receive both vectors and tensors as inputs, while the other models can only receive vectors as inputs. For a fair comparison, in experiments, we compare the proposed SLA-SAN method with SVM, GFK, LAN, ASVM, DTL, STL and TTL, while compare the SLA-SCAN method with CNN, DTL and STL. The convolutional autoencoder com- ponent used in SCAN has the same network structure as the CNN model, except that the fully connected layer is connect- ed to two unpooling layers and two deconvolutional layers to reconstruct inputs. For all deep-learning based models, we use all the training data for pre-training.

自学习（STL）我们对所有中间域数据应用自动编码器或卷积自动编码器以学习通用特征表示，然后使用新特征表示训练具有所有源域和目标域标记数据的分类器。

在基线中，CNN可以接收张量作为输入，DTL和STL可以接收矢量和张量作为输入，而其他模型只能接收矢量作为输入。 为了公平比较，在实验中，我们将提出的SLA-SAN方法与SVM，GFK，LAN，ASVM，DTL，STL和TTL进行比较，同时将SLA-SCAN方法与CNN，DTL和STL进行比较。 SCAN中使用的卷积自动编码器组件具有与CNN模型相同的网络结构，除了完全连接的层连接到两个解开层和两个解卷积层以重建输入。 对于所有基于深度学习的模型，我们使用所有训练数据进行预训练。

### 数据集

The datasets used for experiments include Caltech-256 (Grif- fin, Holub, and Perona 2007) and Animals with Attributes (AwA)3. The Caltech-256 dataset is a collection of 30,607 images from 256 object categories, where the number of in- stances per class varies from 80 to 827. Since we need to transfer knowledge between distant categories, we choose a subset of categories including ‘face’, ‘watch’, ‘airplane’, ‘horse’, ‘gorilla’, ‘billiards’ to form the source and target domains. Specifically, we first randomly choose one cate- gory to form the source domain, and consider images be- longing to this category as positive examples for the source domain. Then we randomly choose another category to form the target domain, and consider images in this category as positive examples for the target domain. As this dataset has a clutter category, we sample images from this category as negative samples for the source and target domains and the sampling process guarantees that each pair of source and tar- get domains has no overlapping on negative samples. After constructing the source domain and the target domain, all the other images in the dataset are used as the intermediate do- mains data. Therefore, we construct P62 = 30 pairs of DDTL problems from this dataset.

用于实验的数据集包括Caltech-256（Grif-fin，Holub和Perona 2007）和Animal with Attributes（AwA）3。 Caltech-256数据集是来自256个对象类别的30,607个图像的集合，其中每个类别的实例数量从80到827不等。由于我们需要在远程类别之间传递知识，我们选择包括'face的类别的子集'，'观察'，'飞机'，'马'，'大猩猩'，'台球'形成源和目标域。具体来说，我们首先随机选择一个类别来形成源域，并考虑将此类别的图像作为源域的正例。然后我们随机选择另一个类别来形成目标域，并将此类别中的图像视为目标域的正例。由于此数据集具有杂波类别，我们将此类别的图像作为源域和目标域的负样本进行采样，并且采样过程保证每对源和目标域在负样本上没有重叠。构建源域和目标域后，数据集中的所有其他图像将用作中间域数据。因此，我们从该数据集构造P62 = 30对DDTL问题。

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机10.JPG)

The AwA dataset consists of 30,475 images of 50 ani- mals classes, where the number of samples per class varies from 92 to 1,168. Due to the copyright reason, this dataset only provides extracted SIFT features for each image in- stead of the original images. We choose three categories including ‘humpback+whale’, ‘zebra’ and ‘collie’ to form the source and target domains. The source-target domains construction procedure is similar to that on the Caltech-256 dataset, and images in the categories ‘beaver’, ‘blue+whale’,
‘mole’, ‘mouse’, ‘ox’, ‘skunk’, and ‘weasel’ are considered as negative samples for the source domain and the target domain. In total we construct P32 = 6 pairs of DDTL problems.

AwA数据集由50个动物类的30,475个图像组成，每个类的样本数从92到1,168不等。 由于版权原因，此数据集仅为每个图像提供提取的SIFT特征，而不是原始图像。 我们选择三个类别，包括'座头鲸+鲸鱼'，'斑马'和'牧羊犬'，以形成源和目标域。 源 - 目标域构建过程类似于Caltech-256数据集上的图像，以及类别'beaver'，'blue + whale'中的图像，'mole'，'mouse'，'ox'，'skunk'和'weasel'被视为源域和目标域的负样本。 总共我们构造了P32 = 6对DDTL问题。

### 性能比较

In experiments, for each target domain, we randomly sample 6 labeled instances for training, and use the rest for testing. Each configuration is repeated 10 times. On the Caltech- 256 dataset, we conduct two sets of experiments. The first experiment uses the original images as inputs to compare SLA-SCAN with CNN, DTL and STL. The second experi- ment uses SIFT features extracted from images as inputs to do a comparison among SLA-SAN and the rest baselines. On the AwA dataset, since only SIFT features are given, we compare SLA-SAN with SVM, GFK, LAN, ASVM, DTL, STL and TTL. The comparison results in terms of average accuracy are shown in Figure 3. From the results, we can see that the transfer learning methods such as DTL, GFK, LAN and ASVM achieve worse performance than CNN or SVM because the source domain and the target domain have huge distribution gap, which leads to ‘negative transfer’. TTL does not work either due to the distribution gap among the source, intermediate and target domains. STL achieves slightly better performance than the supervised learning methods. A rea- son is that the universal feature representation learned from the intermediate domain data helps the learning of the target model to some extent. Our proposed SLA method obtains the best performance under all the settings. We also report the average accuracy as well as standard deviations of different 0.85 methods on some selected tasks in Tables 1 and 2. On these tasks, we can see that for SLA-SCAN, the improvement in accuracy over CNN is larger than 10% and sometimes up to 17%. For SLA-SAN, the improvement in accuracy is around 10% over SVM.

在实验中，对于每个目标域，我们随机抽取6个标记的实例进行训练，并使用其余的进行测试。每个配置重复10次。在Caltech-256数据集上，我们进行了两组实验。第一个实验使用原始图像作为输入来比较SLA-SCAN与CNN，DTL和STL。第二个实验使用从图像中提取的SIFT特征作为输入，以在SLA-SAN和其余基线之间进行比较。在AwA数据集上，由于只给出了SIFT功能，我们将SLA-SAN与SVM，GFK，LAN，ASVM，DTL，STL和TTL进行比较。平均准确度方面的比较结果如图3所示。从结果中我们可以看出，DTL，GFK，LAN和ASVM等传输学习方法的性能比CNN或SVM差，因为源域和目标域有巨大的分配差距，导致“负转移”。由于源域，中间域和目标域之间的分布差距，TTL不起作用。 STL比监督学习方法的性能稍好一些。原因是从中间域数据中学习的通用特征表示有助于在一定程度上学习目标模型。我们提出的SLA方法在所有设置下都获得了最佳性能。我们还报告了表1和表2中某些选定任务的不同0.85方法的平均准确度和标准差。在这些任务中，我们可以看到，对于SLA-SCAN，CNN精度的提高大于10％，有时高达17％。对于SLA-SAN，精度的提高比SVM高约10％。

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机11.JPG)

### 详细结果

To understand how the data in intermediate domains can help connect the source domain and the target domain, in Figure 4, we show some images from intermediate domains selected in different iterations of the SLA algorithm on two transfer learning tasks, ‘face-to-airplane’ and ‘face-to-watch’, on the Caltech-256 dataset. Given the same source domain ‘face’ and two different target domains ‘airplane’ and ‘watch’, in Figure 4, we can see that the model selects completely dif- ferent data points from intermediate domains. It may be not easy to figure out why those images are iteratively selected from our perspective. However, intuitively we can find that at the beginning, the selected images are closer to the source images such as ‘buddhas’ and ‘football-helmet’ in the two tasks, respectively, and at the end of the learning process, the selected images look closer to the target images. Moreover, in Figure 4, we show that as the iterative learning process pro- ceeds, the number of positive samples in the source domain involved decreases and the value of the objective function in problem (3) decreases as well.

为了理解中间域中的数据如何帮助连接源域和目标域，在图4中，我们展示了在SLA算法的不同迭代中选择的中间域中的一些图像，用于两个传输学习任务，“面对面飞机”在Caltech-256数据集上'和'面对面观察'。给定相同的源域'face'和两个不同的目标域'airplane'和'watch'，在图4中，我们可以看到该模型从中间域中选择完全不同的数据点。要弄清楚为什么从我们的角度迭代选择这些图像可能并不容易。然而，直观地我们可以发现，在开始时，所选择的图像分别在两个任务中更接近源图像，例如“buddhas”和“football-helmet”，并且在学习过程结束时，所选图像看得更接近目标图像。此外，在图4中，我们表明，随着迭代学习过程的进行，源域中的正样本数量减少，问题（3）中目标函数的值也减少。

### 不同学习秩序的比较

As shown in Figure 4, the SLA algorithm can learn an or- der of useful intermediate data to be added into the learning process. To compare different orders of intermediate data to be added into learning, we manually design three ordering strategies. In the first strategy, denoted by Random, we ran- domly split intermediate domains data into ten subsets. In each iteration, we add one subset into the learning process. In the second strategy, denoted by Category, we first obtain an order using the SLA algorithm, and then use the additional category information of the intermediate data to add the inter- mediate data category by category into the learning process. Hence the order of the categories is the order they first appear in the order learned by the SLA algorithm. The third strategy, denoted by Reverse, is similar to the second one except that the order of the categories to be added is reversed. In the experiments, the source domain data are removed in the same way as the original learning process of the SLA algorithm. From the results shown in Figure 5, we can see that the three manually designed strategies obtain much worse accuracy than SLA. Furthermore, among the three strategies, ‘Cate- gory’ achieves the best performance, which is because this order generation is close to that of SLA.

如图4所示，SLA算法可以学习将有用的中间数据添加到学习过程中。为了比较要添加到学习中的不同中间数据顺序，我们手动设计了三种排序策略。在第一个策略中，用Random表示，我们随意将中间域数据分成十个子集。在每次迭代中，我们在学习过程中添加一个子集。在第二种策略中，由Category表示，我们首先使用SLA算法获取订单，然后使用中间数据的附加类别信息将中间数据类别按类别添加到学习过程中。因此，类别的顺序是它们首先以SLA算法学习的顺序出现的顺序。第三种策略，由Reverse表示，与第二种策略相似，只是要添加的类别的顺序是相反的。在实验中，源域数据以与SLA算法的原始学习过程相同的方式被移除。从图5所示的结果中，我们可以看到三种手动设计的策略获得的精度比SLA低得多。此外，在三种策略中，“Cateory”实现了最佳性能，这是因为此订单生成与SLA的订单生成接近。

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机13.JPG)

![](http://118.25.176.50/md_img/论文_人脸迁移到飞机12.JPG)

## 结论

In this paper, we study the novel DDTL problem, where the source domain and the target domain are distant, but can be connected via some intermediate domains. To solve the DDTL problem, we propose the SLA algorithm to gradual- ly select unlabeled data from the intermediate domains to bridge the two distant domains. Experiments conducted on two benchmark image datasets demonstrate that SLA is able to achieve state-of-the-art performance in terms of accuracy. As a future direction, we will extend the proposed algorithm to handle multiple source domains.

在本文中，我们研究了新的DDTL问题，其中源域和目标域是远距离的，但可以通过一些中间域连接。 为了解决DDTL问题，我们建议SLA算法逐步从中间域中选择未标记的数据来桥接两个远端域。 在两个基准图像数据集上进行的实验表明，SLA能够在准确性方面实现最先进的性能。 作为未来的方向，我们将扩展所提出的算法以处理多个源域。

## 致谢

We are supported by National Grant Fundamental Research (973 Program) of China under Project 2014CB340304, Hong Kong CERG projects 16211214, 16209715 and 16244616, and Natural Science Foundation of China under Project 61305071. Sinno Jialin Pan thanks the support from the NTU Singapore Nanyang Assistant Professorship (NAP) grant M4081532.020. We thank Hao Wang for helpful discussions and reviewers for their valuable comments.

我们得到了中国国家基金研究（973计划）项目2014CB340304，香港CERG项目16211214,16209715和16244616以及中国自然科学基金项目61305071的支持.Sinno Jialin Pan感谢NTU新加坡南洋助理的支持 教授（NAP）授予M4081532.020。 我们感谢王浩的有益讨论和评论者的宝贵意见。

