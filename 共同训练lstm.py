import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
Staiton = ['s1', 's10', 's12', 's14', 's15', 's4', 's5', 's8']
out = pd.DataFrame()



def load_data(station):
    if station not in Staiton:
        return None
    # 读取站点数据
    ds = pd.read_csv(r'E:\项目\ai技术实践\station.csv')
    # 过滤所需站点数据
    ds = ds[ds['换热站ID'] == station]
    # 时间格式化并以此作为索引
    ds.index = pd.to_datetime(ds.pop('时间'), format='%Y-%m-%d %H:%M:%S')
    # 删除无用字段
    ds.pop('换热站ID')
    ds.pop('一网供水温度')
    ds.pop('一网回水温度')
    ds.pop('一网供水压力')
    ds.pop('一网回水压力')
    ds.pop('瞬时流量')
    # 按照小时重采样（以便与站点数据合并）
    ds = ds.resample('h').mean()
    # print(ds)
    # 读取天气数据
    dw = pd.read_csv(r'weather.csv')
    # 时间格式化并以此作为索引
    dw.index = pd.to_datetime(dw.pop('时间'), format='%Y-%m-%d %H:%M:%S')

    dw['距离'] = station
    change = {"s1": 0.461, "s4": 2.1, "s5": 0.887, "s8": 1.7, "s10": 2.3, "s12": 3.5, "s14": 3.6, 's15': 2.5}
    dw['距离'] = dw['距离'].map(change)
    #  删除无用字段
    dw.pop('日出时间')
    dw.pop('日落时间')
    dw.pop('天气')
    # 合并站点和天气数据
    data = pd.concat([dw, ds], axis=1)
    return data


# 数据读取测试
# data = load_data(station)
# print(data)


class TimeSeriesPredictionDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data.values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length][3]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


#  TimeSeriesPredictionDataset测试
data = load_data('s1')
print(data)
data = TimeSeriesPredictionDataset(data, 24)
print(len(data))
for i in data:
    print(i)

from torch import nn

# 模型定义
device = 'cuda'


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 初始化 LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 初始化一个全连接层，将 LSTM 输出转换为所需输出大小
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 的形状应该是 (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 全连接层前向传播
        out = self.fc(out[:, -1, :])
        return out


# 模型测试
# 定义模型参数
# input_size = 4 # 输入特征的维度
# hidden_size = 50  # LSTM 隐藏层的维度
# num_layers = 2    # LSTM 的层数
# output_size = 1   # 输出的维度

# # 创建模型实例
# model = SimpleLSTM(input_size, hidden_size, num_layers, output_size)

# # 创建一个假的输入数据来演示模型前向传播
# # 输入数据的形状应该是 (batch_size, seq_length, input_size)
# # 例如，batch_size=64, seq_length=20
# dummy_input = torch.rand(64, 20, input_size)
#
# # # # 进行前向传播测试
# # # output = model(dummy_input)
# # # print(output.shape)  # 应该输出 (64, 1)，即批次中每个样本的预测结果
#
import copy
#
# # 设定
batch_size = 20


# 载入数据
data1 = load_data('s1')
data10=load_data('s10')
data12=load_data('s12')
data14=load_data('s14')
data15=load_data('s15')
data4=load_data('s4')
data5=load_data('s5')
data8=load_data('s8')
print(type(data))
data = pd.concat([data1,data4,data5,data8,data10,data12,data14,data15], ignore_index=True)
print(data)
# 分割训练集和测试集
train_size = int(0.7* len(data))
train = data.iloc[:train_size]
test = data.iloc[train_size:]
print('数据集大小为: ', data.shape)
print('训练集的大小为: ', train.shape)
print('验证集大小为: ', test.shape)

# 创建数据集
train_set = TimeSeriesPredictionDataset(train, 24)
train = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_set = TimeSeriesPredictionDataset(test, 24)
test = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# 创建模型
# 定义模型参数
input_size = 4  # 输入特征的维度
hidden_size = 50 # LSTM 隐藏层的维度
num_layers = 2  # LSTM 的层数
output_size = 1  # 输出的维度

# 创建模型实例
model = SimpleLSTM(input_size, hidden_size, num_layers, output_size)
model.to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 开始训练
best_model = None
min_test_loss = float('inf')
train_loss = []
test_loss = []

num_epochs = 100
for epoch in range(num_epochs):
    # 训练
    model.train()
    for x_batch, y_batch in train:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        # 计算网络输出
        output = model(x_batch)
        # 计算损失
        loss = criterion(output, y_batch)
        # print(loss)
        train_loss.append(loss.item())
        # 计算梯度和反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 验证
    model.eval()
    for x_batch, y_batch in test:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            output = model(x_batch)
            loss = criterion(output, y_batch)
            test_loss.append(loss.item())
        # 保存最优模型
        if test_loss[-1] < min_test_loss:
            min_test_loss = test_loss[-1]
            best_model = copy.deepcopy(model)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss[-1]}, Test Loss: {test_loss[-1]}')
model_dir = './model'
model_path = os.path.join(model_dir, 'model.pth')

# 如果目录不存在，则创建它
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 现在目录已存在，可以安全地保存模型
torch.save(best_model, model_path)

# # 载入模型，如在提交文件中使用，需要提前定义模型类SimpleLSTM
model = torch.load(model_path)
model.to(device)
# # 设置模型为评估模式
model.eval()
for m in range(8):
    station=Staiton[m]
    # 获取数据
    # 使用训练集中的最后一组数据作为第一组数据
    ds = pd.read_csv(r'E:\项目\ai技术实践\station.csv')
    ds = ds[ds['换热站ID'] == station]
    ds.index = pd.to_datetime(ds.pop('时间'), format='%Y-%m-%d %H:%M:%S')
    ds.pop('换热站ID')
    ds.pop('一网供水温度')
    ds.pop('一网回水温度')
    ds.pop('一网供水压力')
    ds.pop('一网回水压力')
    ds.pop('瞬时流量')
    ds = ds.resample('h').mean()
    dw = pd.read_csv(r'E:\项目\ai技术实践\weather.csv')
    dw.index = pd.to_datetime(dw.pop('时间'), format='%Y-%m-%d %H:%M:%S')
    dw.pop('日出时间')
    dw.pop('日落时间')
    dw.pop('天气')
    dw['距离'] = station
    change = {"s1": 0.461, "s4": 2.1, "s5": 0.887, "s8": 1.7, "s10": 2.3, "s12": 3.5, "s14": 3.6, 's15': 2.5}
    dw['距离'] = dw['距离'].map(change)
    data = pd.concat([dw, ds], axis=1)
    print(data)
    # 构建推理用数据
    station_dis=[0.461,2.3,3.5,3.6,2.5,2.1,0.887,1.7]
    dw = pd.read_csv(r'E:\项目\ai技术实践\weather_predict.csv')
    dw.index = pd.to_datetime(dw.pop('时间'), format='%Y-%m-%d %H:%M:%S')
    dw['距离'] = station_dis[m]


    dw.pop('日出时间')
    dw.pop('日落时间')
    dw.pop('天气')
    data = pd.concat([data.tail(24), dw])
    print(data)

    # 开始推理
    output = []
    batch = TimeSeriesPredictionDataset(data, 24)
    for i, j in zip(data.index[24:], range(len(batch))):
        d = data[j:j + 24]
        d = torch.tensor(d.values, dtype=torch.float32)
        d = d.unsqueeze(0)
        d=d.to(device)
        o = model(d)
        o = o.item()
        data.loc[i, '瞬时热量'] = o
        # print(data)

    # 整理数据（s1站点）
    data = data.tail(24)
    out_1 = pd.DataFrame()
    out_1['换热站ID'] = station
    out_1['时间'] = data.index
    out_1['换热站ID'] = station
    for i, j in zip(out_1.index, data.index):
        out_1.loc[i, '瞬时热量'] = data.loc[j, '瞬时热量']
    print(out_1)

    # 临时举措！
    # 使用s1站点数据，填充各个站点的数据

    out = pd.concat([out, out_1])

    # 按照换热站ID和时间进行排序（很重要！）
    out = out.sort_values(by=['换热站ID', '时间'])
    print(out)
    # 保存文件

    out.pop('时间')
    print(out)
    out.to_csv('result.csv', header=False)