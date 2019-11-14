import numpy as np
import pandas as pd #读取csv文件的库
import matplotlib.pyplot as plt
import torch
#from torch.autograd import Variable
import torch.optim as optim

class model_build:
    def __init__(self, weights, biases, weights2):
        self.weights = weights
        self.biases = biases
        self.weights2 = weights2
    
    def forward(self, x):
        hidden = x.mm(self.weights) + self.biases.expand(x.size()[0], 10)
        hidden = torch.sigmoid(hidden)
        output = hidden.mm(self.weights2)
        return output

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

# 加载模型
import pickle
model = open('model.pkl', 'rb')
parameters = pickle.load(model)
weights, biases, weights2 = parameters['weights'], parameters['biases'], parameters['weights2']
model.close()
# 加载GPU
cuda = torch.device('cuda')     # Default CUDA device
cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')  # GPU 1 (these are 0-indexed)

x = torch.tensor(features, dtype = torch.double, device=cuda0, requires_grad = True)
y = torch.tensor(targets, dtype = torch.double, device=cuda0, requires_grad = True)

a = model_build(weights, biases, weights2)
output = a.forward(x)

# 还原预测数据
scaled_featuresF = open('scaled_features.pkl', 'rb')
scaled_features = pickle.load(scaled_featuresF)
mean, std = scaled_features['scaled_features']
predictions = (output * std + mean).cpu().data.numpy() 


fig, ax = plt.subplots(figsize = (10, 7))
ax.plot(predictions , label='Prediction', linestyle = '--')
ax.plot(targets * std + mean, label='Data', linestyle = '-')
ax.legend()
ax.set_xlabel('Date-time')
ax.set_ylabel('Counts')
# 对横坐标轴进行标注
plt.savefig('evaluate_predictions_plot.jpg')


