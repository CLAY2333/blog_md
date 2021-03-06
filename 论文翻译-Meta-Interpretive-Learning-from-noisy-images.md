---
title: 论文翻译-Meta-Interpretive_Learning_from_noisy_images
date: 2018-10-13 13:39:26
categories: 论文翻译
tags: [机器学习,图像处理,翻译]
---

<Excerpt in index | 首页摘要> 

第一次翻译论文，因为不是专业的所以在翻译上有什么不足之处还请谅解。这次翻译的论文是期刊《machine learning》上的一篇2018年才发表的关于图像处理的论文。

<!-- more -->

<The rest of contents | 余下全文>

# 从嘈杂的图像中进行元解释学习

## 摘要

统计机器学习被广泛用于图像分类。然而，大多数技术（1）需要许多图像以实现高精度;（2）不提供低于分类水平的推理支持，因此不能支持二次推理，例如光源和图像外的其他物体的存在和位置。本文描述了一种称为逻辑视觉的归纳逻辑编程方法，该方法克服了这些局限性。 LV使用元解释学习（MIL）结合从图像中采样的高对比度点的低级提取来学习描述图像的递归逻辑程序。在已发表的着作中，LV被证明能够高精度地预测类，例如来自少量图像的正多边形，其中支持向量机和卷积神经网络在某些情况下给出接近随机的预测。到目前为止，LV仅适用于无噪声，人工生成的图像。本文通过（a）使用MIL系统Metagol的新的噪声传输版本解决分类噪声来扩展LV，（b）使用原始级统计估计器来识别属性噪声以识别真实图像中的子对象，（c）使用代表经典2D形状（如圆形和椭圆形）的更广泛的背景模型，（d）以简单但通用的光反射循环理论的形式提供更丰富的可学习背景知识。在我们的实验中，我们在自然科学设置和RoboCup竞赛环境中考虑噪声图像。自然科学设置涉及在望远镜和显微镜图像中识别光源的位置，而RoboCup设置涉及识别球的位置。我们的结果表明，对于真实图像，使用单个示例（即单次LV）的新的抗噪声LV版本收敛到至少与三十次统计机器学习者相比的预测隐藏光源中的隐藏光源的精度。科学设置和RoboCup设置。此外，我们证明了光的一般背景递归理论本身可以用LV发明，并用于识别物体的凸起/凹陷中的模糊性，例如科学环境中的陨石坑和RoboCup环境中球的部分遮挡	

**Meta-Interpretive Learning有时候也叫Meta learning，**