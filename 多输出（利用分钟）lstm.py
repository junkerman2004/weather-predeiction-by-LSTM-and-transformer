import numpy as np
import pandas as pd
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import torch
import copy
from torch import nn
import os
Staiton = ['s1', 's4', 's5', 's8', 's10', 's12', 's14', 's15']



if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDA设备对象
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
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
    ds['温差']=ds['一网供水温度'] - ds['一网回水温度']
    ds['换热站距离']=ds['换热站ID']
    change={"s1":0.461,"s4":2.1,"s5":0.887,"s8":1.7,"s10":2.3,"s12":3.5,"s14":3.6,'s15':2.5}
    ds['换热站距离']=ds['换热站距离'].map(change)
    ds.pop('换热站ID')
    ds.pop('一网供水温度')
    ds.pop('一网回水温度')
    ds.pop('一网供水压力')
    ds.pop('一网回水压力')
    # 按照小时重采样（以便与站点数据合并）
    ds = ds.resample('h')['换热站距离','温差','瞬时流量','瞬时热量'].mean()
    #print(ds)
    # 读取天气数据
    dw = pd.read_csv('weather.csv')
    # 时间格式化并以此作为索引
    dw.index = pd.to_datetime(dw.pop('时间'), format='%Y-%m-%d %H:%M:%S')
    #  删除无用字段
    weather_map = {'晴': 1, '阴': 0.7,"多云":0.8,'小雪':0.6,'大雪':0.5}
    dw['日出时间'] = pd.to_datetime(dw['日出时间'], format='%Y-%m-%d %H:%M:%S')
    dw['日落时间'] = pd.to_datetime(dw['日落时间'], format='%Y-%m-%d %H:%M:%S')
    dw['日照时长'] = (dw['日落时间'] - dw['日出时间']).dt.total_seconds() / 3600  # 转换为小时
    dw['天气'] = dw['天气'].map(weather_map).fillna(0.6)
    dw['日照影响']=dw['日照时长']*dw['天气']
    dw.pop('日出时间')
    dw.pop('日落时间')
    dw.pop('日照时长')
    dw.pop('天气')
    #print(dw)
    # 合并站点和天气数据
    data = pd.concat([dw, ds], axis=1)
    return data

# # 数据读取测试
#data = load_data('s1')
#print(data)

#可以实现打包成一个站台一天的内容
class TimeSeriesPredictionDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data.values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length][:,:4]
        y = self.data[idx + self.seq_length-1][4:7]

        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


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

batch_size = 20

station = 's8'
data=load_data(station)
# 分割训练集和测试集
train_size = int(0.8 * len(data))
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
num_layers = 3    # LSTM 的层数
output_size = 3   # 输出的维度

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
        y_batch=y_batch.squeeze(1)
        optimizer.zero_grad()
        # 计算网络输出
        output = model(x_batch)
        output
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
        y_batch=y_batch.squeeze(1)
        with torch.no_grad():
            output = model(x_batch)
            loss = criterion(output, y_batch)
            test_loss.append(loss.item())
    # 保存最优模型
        if test_loss[-1] < min_test_loss:
            min_test_loss = test_loss[-1]
            best_model = copy.deepcopy(model)
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss[-1]}, Test Loss: {test_loss[-1]}')
model_path = './model/model_' + station + '.pth'
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
torch.save(best_model,model_path)


# 载入模型，如在提交文件中使用，需要提前定义模型类SimpleLSTM
model = torch.load(model_path)
model.to(device)
# 设置模型为评估模式
model.eval()

dm= pd.read_csv(r"E:\项目\ai技术实践\weather_predict.csv")
dm.index=pd.to_datetime(dm.pop('时间'), format='%Y-%m-%d %H:%M:%S')
#  删除无用字段
weather_map = {'晴': 1, '阴': 0.7, "多云": 0.8, '小雪': 0.6, '大雪': 0.5}
dm['日出时间'] = pd.to_datetime(dm['日出时间'], format='%Y-%m-%d %H:%M:%S')
dm['日落时间'] = pd.to_datetime(dm['日落时间'], format='%Y-%m-%d %H:%M:%S')
dm['日照时长'] = (dm['日落时间'] - dm['日出时间']).dt.total_seconds() / 3600  # 转换为小时

dm['天气'] = dm['天气'].map(weather_map).fillna(0.6)
dm['日照影响'] = dm['日照时长'] * dm['天气']
dm['距离']=1.7
dm.pop('日出时间')
dm.pop('日落时间')
dm.pop('日照时长')
dm.pop('天气')
dm=np.array(dm)
dm=dm.reshape(1,24,4)
dm=torch.from_numpy(dm)
dm = dm.to(dtype=torch.float32)
print(dm)
om=torch.zeros((1,24,4),dtype=torch.float32)
concatenated_tensor = torch.cat((om,dm), dim=1)
concatenated_tensor=concatenated_tensor.to(device)
output=[]
for i in range(1,25):
    m=concatenated_tensor[:,i:i+24,:]

    m=m.to(device)
    answer=model(m)
    output.append(answer)
output_tensor = torch.stack(output, dim=0)
print(output_tensor)
# 假设model和device已经定义并正确初始化