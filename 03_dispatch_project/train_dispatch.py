#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:37:37 2019

@author: daniel
"""

import numpy as np
import pandas as pd #读取csv文件的库
import matplotlib.pyplot as plt
import torch
#from torch.autograd import Variable
import torch.optim as optim

# 加载测试集
train_features_path = 'train_features.csv'
train_targets_path = 'train_targets.csv'

train_features = pd.read_csv(train_features_path)
train_targets = pd.read_csv(train_targets_path)

# 将数据转化为numpy格式, X (16875, 56)
features = train_features.values
targets = train_targets['cnt'].values
targets = targets.astype(float)
targets = np.reshape(targets, [len(targets), -1])

# 将数据转化为numpy格式
X = train_features.values
Y = train_targets['cnt'].values
Y = Y.astype(float)

Y = np.reshape(Y, [len(Y),1])
losses = []
train_features.head()


"""
    (二). 构建神经网络并进行训练， 分别两种构建方法： a, 手动编写用Tensor运算的神经网络； b, 调用Pytorch现有函数，构建序列化神经网络
"""

"""
a, 手写用Tensor运算的神经网路
    已知的数据：X.shape[]=(16875, 56), Y.shape = (16875, 1)
    构建网络的参数：
                input_size = X.shape[1],也就是一组特征向量的纬度，这里是56.
                hidden_size = 10, 设置10隐含节点
                output_size = 1, 1个输出节点
                batch_size = 128, 设置批处理大小为 128组向量
                则(三层神经网络：输入，隐含，输出)：
                    weights.shape = (56, 10)
                    biases.shape = (,10)
                    weights2.shape = (10, 1)
"""


cuda = torch.device('cuda')     # Default CUDA device
cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')  # GPU 1 (these are 0-indexed)

input_size = X.shape[1]
hidden_size = 10
output_size = 1
batch_size = 128

weights = torch.randn([input_size, hidden_size], dtype = torch.double, device=cuda0, requires_grad = True)

biases = torch.randn([hidden_size], dtype = torch.double, device=cuda0, requires_grad = True)

weights2 = torch.randn([hidden_size, output_size], dtype = torch.double, device=cuda0, requires_grad = True)


def neu(x):
    hidden = x.mm(weights) + biases.expand(x.size()[0], hidden_size)
    hidden = torch.sigmoid(hidden)
    output = hidden.mm(weights2)
    return output

def cost(x, y):
    error = torch.mean((x-y)**2)
    return error

def optimizer_step(learning_rate):
    weights.data.add_(-learning_rate * weights.grad.data)
    biases.data.add_(-learning_rate * biases.grad.data)
    weights2.data.add_(-learning_rate * weights2.grad.data)
    
def zero_grad():
    if weights.grad is not None and biases.grad is not None and weights2.grad is not None:
        weights.grad.data.zero_()
        biases.grad.data.zero_()
        weights2.grad.data.zero_()

# 神经网络的训练

losses = []


for i in range(1000):
    batch_loss = []
    # start 和 end 分别是提取一个batch的起始和终止下标
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        # 将 X 的每一个batch tensor化
        xx = torch.tensor(X[start:end], dtype = torch.double, device=cuda0, requires_grad = True)
 
        yy = torch.tensor(Y[start:end], dtype = torch.double, device=cuda0, requires_grad = True)

        predict = neu(xx)
        loss = cost(predict, yy)
        loss.backward()
        optimizer_step(0.01)      
        zero_grad()
        batch_loss.append(loss.cpu().data.numpy())
    # 每隔100步输出一下损失值
    if i % 100 ==0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))
        batch_loss =[]

# 打印输出损失值
fig = plt.figure(figsize = (10, 7))
plt.plot(np.arange(len(losses)) * 100, losses, 'o--')
plt.xlabel('epoch')
plt.ylabel('MSE')

import model_function as mf

parameters = {}
parameters['weights'] = weights
parameters['biases'] = biases
parameters['weights2'] = weights2
mf.save_model(parameters, 'model.pkl')