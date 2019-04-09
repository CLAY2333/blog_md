---
title: 吴恩达的机器学习-ex2
date: 2018-09-17 15:10:58
categories: 机器学习
tags: [机器学习,Logistic,正则化]
---

<Excerpt in index | 首页摘要> 

这是第二次的的机器学习编程训练，在本练习中，将实施逻辑回归并将其应用于两个不同的数据集。



<!-- more -->

<The rest of contents | 余下全文>

## 热身运动：双曲线函数

在开始实际成本函数之前，请回想一下逻辑回归假设定义为：hθ(x) = g(θ^T x)

其中函数g是sigmoid函数。 sigmoid函数定义为：g(z) = 1/(1+e^-z)=1/(1+e^-(θ^T x))	

第一步是在sigmoid.m中实现此功能，以便程序的其余部分可以调用它。完成后，尝试通过在Octave / MATLAB命令行中调用sigmoid（x）来测试几个值。对于x的大正值，sigmoid应该接近1，而对于大的负值，sigmoid应该接近0.评估sigmoid（0）应该给出正好0.5。您的代码也应该使用向量和矩阵。对于矩阵，函数应该对每个元素执行sigmoid函数。

### sigmoid.m

~~~octave
function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.
%
% You need to return the following variables correctly 
g = zeros(size(z));
%
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
%
g=1 ./ (1+exp(-z));
%
% =============================================================
%
end
%
~~~

直接依据原型函数写出即可。

## 成本函数和梯度

现在，您将实现逻辑回归的成本函数和梯度。 完成costFunction.m中的代码以返回成本和渐变。回想一下逻辑回归中的成本函数是：

![ex2-1](http://www.lzyclay.cn/md_img/ex2-1.png)

并且成本的梯度是与θ相同长度的向量，其中j^th  element（对于j = 0,1，...，n）定义如下：

![ex2-2](http://www.lzyclay.cn/md_img/ex2-2.png)	

请注意，虽然此渐变看起来与线性回归梯度相同，但公式实际上是不同的，因为线性和逻辑回归具有不同的hθ（x）定义。完成后，ex2.m将使用θ的初始参数调用costFunction。 应该看到成本约为0.693

### costFunction.m

~~~octave
function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
%
% Initialize some useful values
m = length(y); % number of training examples
%
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%
%
%
%
now_g=1 ./ (1+exp(-(X*theta)));
now_J=sum(sum(-y .*log(now_g))-(sum((1-y).*log(1-now_g))));
J=now_J/m;
grad=(sum((now_g-y).*X))/m;
%
%
%
%J=sum(-y'*log(now_g)-(1-y)'*log(1-now_g))/m; %矢量化实现
%grad=((now_g-y)'*X)/m; %矢量化实现
%
%
%
%
% =============================================================
%
end
~~~

##使用fminunc学习参数

  ~~~octave
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
%
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
  ~~~

在此代码段中，我们首先定义了与fminunc一起使用的选项。具体来说，我们将GradObj选项设置为on，它告诉fminunc我们的函数返回成本和渐变。这允许fminunc在最小化函数时使用渐变。此外，我们将MaxIter选项设置为400，以便fminunc在终止之前最多运行400步。为了指定我们最小化的实际函数，我们使用“short-hand”来指定带有@（t）（costFunction（t，X，y））的函数。这将创建一个带有参数t的函数，该函数调用costFunction。这允许我们包装costFunction以用于fminunc。
如果已正确完成costFunction，fminunc将收敛于正确的优化参数并返回成本和θ的最终值。请注意，通过使用fminunc，不必自己编写任何循环，也不必设置像渐变下降那样的学习速率。这完全由fminunc完成：您只需要提供一个计算成本和梯度的函数。
fminunc完成后，ex2.m将使用θ的最佳参数调用costFunction函数。应该看到成本约为0.203。

##评估逻辑回归

在本练习的这一部分中，您将实施正则化逻辑回归，以预测来自制造工厂的微芯片是否通过质量保证（QA）。 在QA期间，每个微芯片都经过各种测试以确保其正常运行。
假如现在工厂的产品经理，在两个不同的测试中获得了一些微芯片的测试结果。 从这两个测试中，想确定是应该接受还是拒绝微芯片。 为了做出决定，可以在过去的微芯片上获得测试结果的数据集，从中可以构建逻辑回归模型。

###costFunctionReg.m

~~~octave
function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));=
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
now_g=1 ./ (1+exp(-(X*theta)));
now_J=sum(sum(-y .*log(now_g))-(sum((1-y).*log(1-now_g))));
now_t = theta(2:length(theta),1);
J=now_J/m+(lambda/(2*m))*sum(power(now_t,2));
%=============================================================
grad=((sum((now_g-y).*X))/m)';
tmp = grad + (lambda/m)*theta;
grad = [grad(1,1);tmp(2:length(theta),1)];
% =============================================================
end
~~~

上述代码的函数原型为下图,下图三个函数就是为了防止函数过度拟合的正则化法。

![ex2-2](http://www.lzyclay.cn/md_img/ex2-3.png)

![ex2-2](http://www.lzyclay.cn/md_img/ex2-4.png)

![ex2-2](http://www.lzyclay.cn/md_img/ex2-5.png)

## 结果

测试了不同的λ对结果的影响

λ=1

![ex2-2](http://www.lzyclay.cn/md_img/ex2-6.png)

λ=0

![ex2-2](http://www.lzyclay.cn/md_img/ex2-7.png)

λ=100

![ex2-2](http://www.lzyclay.cn/md_img/ex2-8.png)

可以看得出来，在λ过小的时候，函数会出现过度拟合的情况。从而造成精度的缺失。反而是当λ大一些的时候其实精度也并不会很低。所以我们在取值的时候不应该取过低的λ。

## 其他的函数

### plotData.m

~~~octave
function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
% Create New Figure
figure; hold on;
% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
%Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2,'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
% =========================================================================
hold off;
end
~~~

### predict.m

~~~octave
function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
m = size(X, 1); % Number of training examples
% You need to return the following variables correctly
p = zeros(m, 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
p = sigmoid(X * theta)>=0.5;
% =========================================================================
end
~~~

