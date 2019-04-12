---
title: 小试kaggle-Mnist手写数字识别
date: 2019-04-11 16:39:35
categories:
tags:
---

<Excerpt in index | 首页摘要> 

简要的记录了我第一刷kaggle的MNIST比赛过程，成绩为0.98457，排名在1470，基本还算理想，因为网络层比较简单，只采用了两层卷积和三层连接，优化函数为adm，损失函数为BCE。

<!-- more -->

<The rest of contents | 余下全文>

# 定义网络

我设置的网络是两层卷积和三层的全连接，代码如下：

~~~python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(256,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
~~~

# 读取数据

因为kaggle的数据是csv格式的，所以我们使用pandas去读取，这一部分参考了官网的模板，可以根据自己的需求修改：

~~~python
class MNISTCSVDataset(data.Dataset):

    def __init__(self, csv_file, Train=True):
        self.dataframe = pd.read_csv(csv_file, iterator=True)
        self.Train = Train

    def __len__(self):
        if self.Train:
            return 42000
        else:
            return 28000

    def __getitem__(self, idx):
        data = self.dataframe.get_chunk(100)
        ylabel = data['label'].values.astype('float')
        xdata = data.ix[:, 1:].values.astype('float')
        return ylabel, xdata
~~~

# 定义参数

如下：

~~~python
path='data/'#数据集文件位置
batch_size = 1#批次大小
num_epochs = 10#循环次数
learning_rate = 0.01#学习率
ngpu=2
~~~

# 主函数

## 定义网络

~~~python
 net = Net()
 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#交由GPu运算
 if(torch.cuda.device_count()>1):#多块GPU并行计算加速
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model=nn.DataParallel(model,device_ids=[0,1])
~~~

## 损失函数和优化函数

### 损失函数

损失函数，又叫目标函数，是编译一个神经网络模型必须的两个参数之一。另一个必不可少的参数是优化器。损失函数是指用于计算标签值和预测值之间差异的函数，在机器学习过程中，有多种损失函数可供选择，典型的有距离向量，绝对值向量等。一般pytorch中可用的loss函数有：

~~~python
criterion = nn.L1Loss()#最为简单，取误差的平均数即可
criterion = nn.SmoothL1Loss()#误差在 (-1,1) 上是平方损失，其他情况是 L1 损失
criterion = nn.MSELoss()#平方损失函数。其计算公式是预测值和真实值之间的平方和的平均数。
criterion = nn.BCELoss()#二分类用的交叉熵，其计算公式较复杂，这里主要是有个概念即可，一般情况下不会用到。
criterion = nn.CrossEntropyLoss()#交叉熵损失函数
criterion = F.nll_loss()#负对数似然损失函数（Negative Log Likelihood）
criterion = nn.NLLLoss2d()#和上面类似，但是多了几个维度，一般用在图片上。
~~~

我们这次主要采用的是交叉熵损失函数:

~~~python
criterion = nn.CrossEntropyLoss()
~~~

### 优化函数

越复杂的神经网络 , 越多的数据 , 我们需要在训练神经网络的过程上花费的时间也就越多. 原因很简单, 就是因为计算量太大了. 可是往往有时候为了解决复杂的问题, 复杂的结构和大数据又是不能避免的, 所以我们需要寻找一些方法, 让神经网络聪明起来, 快起来。那么这就是优化函数的作用啦，一般有以下几种：

~~~python
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)#随机梯度下降
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)#Momentum 更新方法
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)#RMSProp 更新方法
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))#AdaGrad 更新方法
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
~~~

我们这里采用Adam：

~~~python
optimizer = torch.optim.Adam(net.parameters(),lr=0.001,betas=(0.9, 0.99))
~~~

## 训练过程

这里最重要的就要来了，那就是训练过程。直接代码：

~~~python
for epochs in range(num_epochs):
    mydataset = MNISTCSVDataset(path + "train.csv")  # 读取数据集
	train_loader = torch.utils.data.DataLoader(mydataset, batch_size=batch_size, shuffle=True)
	for step, (lable, data) in enumerate(train_loader):
	data=data.to(device)#传入GPU
	lable=lable.to(device)#传入GPU
	output = net(data.view(100, 1, 28, 28).float())#进行网络计算
	lable = lable.long()#转类型
	loss=criterion(output,lable.squeeze())#计算损失函数
	optimizer.zero_grad()#梯度至零
	loss.backward()#损失回传
	optimizer.step()#计算梯度
	if step % 20 == 0:
		print('epochs:',epochs,'step %d' % step, loss)
~~~

 

# 总结

以上就是我简单的写得kaggle上的MNIST手写数字的CNN代码了，很简单也有许多需要优化的地方，日后再优化吧，一下为总的代码：
main.py:

~~~python
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import csv
import pandas as pd
import tqdm as tqdm

path='data/'#数据集文件位置
batch_size = 1#批次大小
num_epochs = 10#循环次数
learning_rate = 0.01#学习率
ngpu=2

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(256,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MNISTCSVDataset(data.Dataset):

    def __init__(self, csv_file, Train=True):
        self.dataframe = pd.read_csv(csv_file, iterator=True)
        self.Train = Train

    def __len__(self):
        if self.Train:
            return 42000
        else:
            return 28000

    def __getitem__(self, idx):
        data = self.dataframe.get_chunk(100)
        ylabel = data['label'].values.astype('float')
        xdata = data.ix[:, 1:].values.astype('float')
        return ylabel, xdata


if __name__ == '__main__':
    net = Net()
    print(net)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 开始读取数据
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001,betas=(0.9, 0.99))
    for epochs in range(num_epochs):
        mydataset = MNISTCSVDataset(path + "train.csv")  # 读取数据集
        train_loader = torch.utils.data.DataLoader(mydataset, batch_size=batch_size, shuffle=True)
        for step, (lable, data) in enumerate(train_loader):
            data=data.to(device)
            lable=lable.to(device)
            output = net(data.view(100, 1, 28, 28).float())
            lable = lable.long()
            loss=criterion(output,lable.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print('epochs:',epochs,'step %d' % step, loss)

    torch.save(net, 'divided-net.pkl')

~~~

test.py:

~~~python
import torch
import torch.utils.data as data
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import csv
file = 'data/test.csv'

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(256,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MNISTCSVDataset(data.Dataset):

    def __init__(self, csv_file, Train=False):
        self.dataframe = pd.read_csv(csv_file, iterator=True)
        self.Train = Train

    def __len__(self):
        if self.Train:
            return 42000
        else:
            return 28000

    def __getitem__(self, idx):
        data = self.dataframe.get_chunk(100)
        xdata = data.as_matrix().astype('float')
        return xdata


net = torch.load('divided-net.pkl', map_location=lambda storage, loc: storage.cuda(0))


myMnist = MNISTCSVDataset(file)
test_loader = torch.utils.data.DataLoader(myMnist, batch_size=1, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

values = []
for _, xd in enumerate(test_loader):
    xd = xd.to(device)
    output = net(xd.view(100, 1, 28, 28).float())
    values = values + output.argmax(dim=1).cpu().numpy().tolist()

with open('data/sample_submission.csv', 'r') as fp_in, open('newfile.csv', 'w', newline='') as fp_out:
    reader = csv.reader(fp_in)
    writer = csv.writer(fp_out)
    header = 0
    for i, row in enumerate(reader):
        if i == 0:
            writer.writerow(row)
        else:
            row[-1] = str(values[i-1])
            writer.writerow(row)
~~~

