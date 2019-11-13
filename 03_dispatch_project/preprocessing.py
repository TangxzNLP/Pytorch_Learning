import numpy as np
import pandas as pd #读取csv文件的库
import matplotlib.pyplot as plt
import torch
#from torch.autograd import Variable
import torch.optim as optim

data_path = 'bike-sharing-dataset/hour.csv'
rides = pd.read_csv(data_path)
rides.head()

"""
    (一). 数据预处理，包括：a,类型变量 one-hot化, 剔除无用特征数据以及多余特征数据；b, 数值变量标准化； c.对数据进行train, test划分
"""

"""
a 类型变量: 有 season, mnth, hr, weekday, weathersit(天气情况),这5个变量的数值的递增并不对应信号强度的增大，所以要做one-hot化处理，
这里会用到 panda中的pd.get_dummies(), 如dummy = pd.get_dummies(rides['season'], prefix = season, drop_first = false)
使得season一列变成season1, season2, season3, season4, 再通过pd.concat([rides, dummy], axis = 1) 将one-hot化的内容加入到
rides中，但是season 会被保留下来，再通过 rides.drop('season', axis = 1)将该列剔除. 
season ranges from 1~4
mnth   ranges from 1~12
hr     ranges from 1~24
weekday ranges from 0~6
weathersit ranges from 1~4
"""
#即将要one-hot处理的列，label名
dummy_fields = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix = each, drop_first = False)
    rides = pd.concat([rides, dummies], axis = 1)


# 将原有类型变量对应的列去掉，同时把一些不相关的列（特征）去掉
fields_to_drop = ['season', 'mnth', 'hr', 'weekday', 'weathersit', 'instant', 'dteday', 'atemp', 'workingday']
data = rides.drop(fields_to_drop, axis = 1)
# 查看现在信息
data.head()

"""
b 数值变量: 有 cnt, temp, hum, windspeed. 因为每个变量都是相互独立的，所以他们绝对值大小与问题本身没有关系，为了消除数值之间的差异，我们
对每一个数值变量进行标准化处理。使其数值在0左右波动。比如，temp，它在整个数据库中取值mean(temp), 方差为std(temp).
"""
quant_features = ['cnt', 'temp', 'hum', 'windspeed']
# 将每个变量的均值和方差都存储到scaled_features变量中
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] =(data[each]-mean)/std

# 保存scaled_features, 在evaluate将数据还原出来
import pickle
F = open('scaled_features.pkl','wb')
parameters = {}
parameters['scaled_features'] = scaled_features['cnt']
pickle.dump(parameters, F) 
F.close()

"""
c 对数据集进行分割， 将所有的数据集分为测试集和训练集。 将以后21天数据一共 21*24个数据点作为测试集，其它是训练集
"""
test_data = data[-21*24:]
train_data = data[:-21*24]
#print("训练数据：", len(train_data), '测试数据：', len(test_data))

# 将数据划分为特征列，与目标列
target_fields = ['cnt', 'casual', 'registered']
train_features, train_targets = train_data.drop(target_fields, axis = 1), train_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis = 1), test_data[target_fields]

print(train_features.shape, type(train_features))
# 保存分别保存 train_features, train_targets, test_features, test_targets

train_features.to_csv('train_features.csv', index = None)
train_targets.to_csv('train_targets.csv', index = None)
test_features.to_csv('test_features.csv', index = None)
test_targets.to_csv('test_targets.csv', index = None)

