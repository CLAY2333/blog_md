---
title: 吴恩达的机器学习-ex1
date: 2018-09-17 10:41:24
categories: 机器学习
tags: [Octave,编程,机器学习,梯度下降,正规方程]
---

<Excerpt in index | 首页摘要> 

是关于机器学习的ex1的一些记录，希望可以帮到你。

<!-- more -->

<The rest of contents | 余下全文>

# ex1.m

~~~octave
%% Machine Learning Online Class - Exercise 1: Linear Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%
%
%% Initialization
clear ; close all; clc
%
%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');
warmUpExercise()
%
fprintf('Program paused. Press enter to continue.\n');
pause;
%
%
%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples
%
% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);
%
fprintf('Program paused. Press enter to continue.\n');
pause;
%
%% =================== Part 3: Cost and Gradient descent ===================
%
X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
%
% Some gradient descent settings
iterations = 1500;
alpha = 0.01;
%
fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');
%
% further testing of the cost function
J = computeCost(X, y, [-1 ; 2]);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n');
%
fprintf('Program paused. Press enter to continue.\n');
pause;
%
fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);
%
% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');
%
% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure
%
% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);
%
fprintf('Program paused. Press enter to continue.\n');
pause;
%
%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')
%
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
%
% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));
%
% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end
%
%
% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');
%
% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
%
~~~

该函数即为主函数，调子函数和画图都是在主函数进行的。这部分我们并没有太多需要更改的地方。接下来将依次介绍各个子函数。

## computeCost.m

~~~octave
function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
%
% Initialize some useful values
m = length(y); % number of training examples
%
% You need to return the following variables correctly 
J = 0;
%
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%
J = sum(power(X*theta-y,2)/(2*m));
%
%
% =========================================================================
%
end
~~~

此函数包含的是单变量代价函数的计算。函数原型为下面的图片的Cost function。power函数第一个参数为幂的底，第二个人为几次方。基本上这个函数就是按照原型写过来的，所以没有什么好讲的。

![image-20180906155604789](http://www.lzyclay.cn/md_img/image-20180906155604789.png)

## gradientDescent.m

~~~octave
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
%
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%
for iter = 1:num_iters
%
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
theta_1=theta(1,1)-alpha/m*sum(X*theta-y);
theta_2=theta(2,1)-alpha/m*sum((X*theta-y) .* X(:,2));
theta(1,1)=theta_1;
theta(2,1)=theta_2;
%
%theta = theta - alpha * (X' * (X * theta - y)) / m;
%
%
    % ============================================================
%
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
%
end
%
end
%
~~~

此函数为梯度下降函数，函数原型为下图的单变量梯度算法函数原型。累加的步骤我们都使用了sum函数求和。而.*的含义是指单纯的只是每一项的相乘。而我们假设两个theta的原因是因为我们需要同步更新，所以用两个临时变量先存储着在进行同步赋值。

![image-20180906155604789](http://www.lzyclay.cn/md_img/梯度算法5.png)

## 结果

训练结果

![image-20180906155604789](http://www.lzyclay.cn/md_img/ex1-run1.png)



![image-20180906155604789](http://www.lzyclay.cn/md_img/ex1-run2.png)

![image-20180906155604789](http://www.lzyclay.cn/md_img/ex1-run3.png)

可以看出来最后最后函数是拟合了的，也到达了全局最优解。证明我们写的单变量梯度下降算法是成功的。接下来我们要尝试的是多变量梯度下降算法。

# ex1_multi.m

~~~octave
%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%
%
%% Initialization
%
%% ================ Part 1: Feature Normalization ================
%
%% Clear and Close Figures
clear ; close all; clc
%
fprintf('Loading data ...\n');
%
%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
%
% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
%
fprintf('Program paused. Press enter to continue.\n');
pause;
%
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);
% Add intercept term to X
X = [ones(m, 1) X];
%
%
%% ================ Part 2: Gradient Descent ================
%
% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%
%
fprintf('Running gradient descent ...\n');
%
% Choose some alpha value
alpha = 0.03;
num_iters = 160;
%
% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
%
% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
%
% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');
%
	% Estimate the price of a 1650 sq-ft, 3 br house
	% ====================== YOUR CODE HERE ======================
	% Recall that the first column of X is all-ones. Thus, it does
	% not need to be normalized.
price = [1,(1650-mu(1))./sigma(1),(3-mu(2))./sigma(2)]*theta; % You should change this
%
%
%
% ============================================================
%
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
%
fprintf('Program paused. Press enter to continue.\n');
pause;
%
%% ================ Part 3: Normal Equations ================
%
fprintf('Solving with normal equations...\n');
%
% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%
%
%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
%
% Add intercept term to X
X = [ones(m, 1) X];
%
% Calculate the parameters from the normal equation
theta = normalEqn(X, y);
%
% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');
%
%
% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = [1,1650,3]*theta; % You should change this
%
%
% ============================================================
%
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);
%
plot(theta(1), theta(2),theta(3),'rx', 'MarkerSize', 10, 'LineWidth', 2);
~~~

和单变量的一样，主函数主要是做的是调用子函数和绘图工作。但是我们需要注意的是这个计算机，我在接下来的例子中将会讲述。

~~~octave
price = [1,(1650-mu(1))./sigma(1),(3-mu(2))./sigma(2)]*theta; 
~~~



## computeCostMulti.m

~~~octave
function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
%
% Initialize some useful values
m = length(y); % number of training examples
%
% You need to return the following variables correctly 
J = 0;
%
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
%
J = sum(power(X*theta-y,2)/(2*m));
%
%
% =========================================================================
%
end
~~~

我们可以发现其实单变量的代价函数计算和多变量的是一样的，并没有什么区别。这里就不多讲了

## featureNormalize.m

~~~octave
function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%
% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
%
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
%
%
%
%
mu = mean(X);
sigma = std(X, 1, 1);
%sigma=max(X)-min(X);
for i = 1 : size(X, 2)
    X_norm(:, i) = (X(:, i) - mu(i)) ./ sigma(i);
end
%
%
%
% ============================================================
%
end
%
~~~

这个函数的主要作用是对训练集进行缩放，因为目前的训练集太大了所以需要对训练集的数字进行缩放。但是需要注意的是，因为我们对训练集进行了缩放，所以在预测值的时候，我们需要对预测的值也进行一个标准化。

## gradientDescentMulti.m

~~~octave
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
%
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%
theta_t=theta;
for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
%
    %怪异的写法
    %theta = theta - alpha / m * X' * (X * theta - y);
    for i=1:length(theta)
    theta_t(i,1)=theta(i,1)-alpha/m*sum((X*theta-y) .* X(:,i));
    end
    theta=theta_t;
%
    % ============================================================
%
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end
%
end
%
~~~

这个函数计算了多变量的梯度下降，用theta_t存储了临时的theta的变量来同步更新。

## normalEqn.m

~~~octave
function [theta] = NORMALEQN(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.
%
theta = zeros(size(X, 2), 1);
%
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%
%
% ---------------------- Sample Solution ----------------------
%
%
theta=pinv(X'*X)*X' *y;
%
% -------------------------------------------------------------
%
%
% ============================================================
%
end
%
~~~

这是使用了正规方程的办法直接计算出了最佳的theta值，函数原型为下：

![image-20180906155604789](http://www.lzyclay.cn/md_img/正规方程1.png)

此方法优点是快，不需要去一步一步的进行梯度下降的算法。但是在一些复杂的地方没办法使用。而且在数量级特别大了的时候

## 结果

~~~
Theta computed from gradient descent:
 337809.605529
 102503.557494
 317.821939
--------------------------------------------------------------------
Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):
 $292016.665614
Program paused. Press enter to continue.
Solving with normal equations...
Theta computed from the normal equations:
 89597.909544
 139.210674
 -8738.019113
--------------------------------------------------------------------
Predicted price of a 1650 sq-ft, 3 br house (using normal equations):
 $293081.464335
~~~

![image-20180906155604789](http://www.lzyclay.cn/md_img/ex1_multi_1.png)

如图所示可以看出，代价函数是在下降的

