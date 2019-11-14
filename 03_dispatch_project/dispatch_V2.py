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

Y = np.reshape(Y, [len(Y),-1])
losses = []
train_features.head()

# 使用Pytorch现成的函数，构建序列化的神经网络
input_size = features.shape[1]
hidden_size = 10
output_size = 1
batch_size = 128

# 根据以上信息，torch.nn.Sequential中调用函数，会自动生成对应的权重和偏置
neu = torch.nn.Sequential(
    # 内在函数生成对应的weights, biases, weights2, 并分别执行hidden运算, sigmoid运算, 以及output运算
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)

# 使用自定义MSELoss函数，usage: cost(predict, y)
cost = torch.nn.MSELoss()

# 梯度计算，采用SGD梯度下降方法
optimizer = torch.optim.SGD(neu.parameters(), lr = 0.01)

# 神经网络训练循环
losses = []
for i in range(1000):
    # 每128个样本点被划分为一嘬, 在循环的时候一批一批的读取 (128, 56)
    batch_loss = []
    # start和end分别是提取一个batch数据的起始和终止下标
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = torch.tensor(X[start:end], dtype = torch.float, requires_grad = True)
        yy = torch.tensor(Y[start:end], dtype = torch.float, requires_grad = True)
        predict = neu(xx)
        loss = cost(predict, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# 保存模型
#import model_function as mf
#mf.save_model(neu.parameters(), 'neu.pkl')
torch.save(neu, 'neu.pkl')
print(neu.state_dict())

fig = plt.figure(figsize=(10, 7))
plt.plot(np.arange(len(losses))*100,losses, 'o-')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.savefig('v2.jpg')

# 测试模型
# 用训练好的神经网络在测试集上进行预测
# 加载测试集
test_features_path = 'test_features.csv'
test_targets_path = 'test_targets.csv'

test_features = pd.read_csv(test_features_path)
test_targets = pd.read_csv(test_targets_path)

# 将数据转化为numpy格式, X (504, 56)
features = test_features.values
targets = test_targets['cnt'].values
targets = targets.astype(float)
targets = np.reshape(targets, [len(targets), -1])


# 将属性和预测变量包裹在Variable型变量中
x = torch.tensor(test_features.values, dtype = torch.float, requires_grad = True)
y = torch.tensor(targets, dtype = torch.float, requires_grad = True)

# 用神经网络进行预测
predict = neu(x)
predict = predict.data.numpy()






