# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import xlsxwriter
workbook = xlsxwriter.Workbook('new.xlsx')
worksheet = workbook.add_worksheet()
# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 输入的历史look_back步，和预测未来的T步
look_back = 5
T = 1
epochs = 50  # 迭代次数
num_features = 5  # 输入特证数
embed_dim = 32  # 嵌入维度
dense_dim = 64  # 隐藏层神经元个数
num_heads = 4  # 头数
dropout_rate = 0.01  # 失活率
num_blocks = 3  # 编码器解码器数
learn_rate = 0.001  # 学习率
batch_size = 24  # 批大小

# 读取数据
# dataset = pd.read_excel('tcndata.xlsx', usecols=[4, 5, 6, 7, 8, 9])
# dataX = dataset.values
# dataY = dataset['功率'].values
# print(dataX)
# print(dataY)
Staiton = ['s1', 's10', 's12', 's14', 's15', 's4', 's5', 's8']
for sta in range(8):
    station=Staiton[sta]

    def load_data(station):
        if station not in Staiton:
            return None
        # 读取站点数据
        ds = pd.read_csv(r"E:\project\ai技术实践\station.csv")
        # 过滤所需站点数据
        ds = ds[ds['换热站ID'] == station]
        # 时间格式化并以此作为索引
        ds.index = pd.to_datetime(ds.pop('时间'), format='%Y-%m-%d %H:%M:%S')
        # 删除无用字段

        ds['换热站距离']=ds['换热站ID']
        change={"s1":0.461,"s4":2.1,"s5":0.887,"s8":1.7,"s10":2.3,"s12":3.5,"s14":3.6,'s15':2.5}
        ds['换热站距离']=ds['换热站距离'].map(change)
        ds.pop(("瞬时流量"))
        ds.pop('换热站ID')
        ds.pop('一网供水温度')
        ds.pop('一网回水温度')
        ds.pop('一网供水压力')
        ds.pop('一网回水压力')
        # 按照小时重采样（以便与站点数据合并）
        ds = ds.resample('h')['换热站距离','瞬时热量'].mean()
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
    def load_predict_data(station):
        dw = pd.read_csv('weather.csv')
        # 时间格式化并以此作为索引
        dw.index = pd.to_datetime(dw.pop('时间'), format='%Y-%m-%d %H:%M:%S')
        #  删除无用字段
        weather_map = {'晴': 1, '阴': 0.7, "多云": 0.8, '小雪': 0.6, '大雪': 0.5}
        dw['日出时间'] = pd.to_datetime(dw['日出时间'], format='%Y-%m-%d %H:%M:%S')
        dw['日落时间'] = pd.to_datetime(dw['日落时间'], format='%Y-%m-%d %H:%M:%S')
        dw['日照时长'] = (dw['日落时间'] - dw['日出时间']).dt.total_seconds() / 3600  # 转换为小时
        dw['天气'] = dw['天气'].map(weather_map).fillna(0.6)
        dw['日照影响'] = dw['日照时长'] * dw['天气']
        dw['距离']=station

        change = {"s1": 0.461, "s4": 2.1, "s5": 0.887, "s8": 1.7, "s10": 2.3, "s12": 3.5, "s14": 3.6, 's15': 2.5}
        dw['距离']=dw['距离'].map(change)
        dw.pop('日出时间')
        dw.pop('日落时间')
        dw.pop('日照时长')
        dw.pop('天气')
        return dw


    data=load_data(station)
    predictdata=load_predict_data(station)

    predictdata=predictdata.values
    dataX=data.values
    dataY=data['瞬时热量'].values
    # print(dataX)
    # print(dataY)
    #归一化数据
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    scaler3=MinMaxScaler(feature_range=(0,1))
    data_X = scaler1.fit_transform(dataX)
    data_Y = scaler2.fit_transform(dataY.reshape(-1, 1))
    predict_data=scaler3.fit_transform(predictdata)

    # 划分训练集和测试集，用80%作为训练集，20%作为验证集
    train_size = int(len(data_X) * 0.8)
    val_size = int(len(data_X) * 0.2)


    train_X, train_Y = data_X[0:train_size], data_Y[0:train_size]
    val_X, val_Y = data_X[train_size:], data_Y[train_size:]
    # print(train_X.size)
    # print(train_Y.size)
    # print(val_X.shape)
    # print(val_Y.size)
    # 定义输入数据，输出标签数据的格式的函数，并将数据转换为模型可接受的3D格式
    def create_dataset(datasetX, datasetY, look_back=1, T=1):
        dataX, dataY = [], []
        for i in range(0, len(datasetX) - look_back - T, T):
            a = datasetX[i:(i + look_back), :]
            dataX.append(a)
            if T == 1:
                dataY.append(datasetY[i + look_back])
            else:
                dataY.append(datasetY[i + look_back:i + look_back + T, 0])
        return np.array(dataX), np.array(dataY)


    # 准备训练集和测试集的数据
    trainX, trainY = create_dataset(train_X, train_Y, look_back, T)
    valX, valY = create_dataset(val_X, val_Y, look_back, T)


    # 转换为PyTorch的Tensor数据
    trainX = torch.Tensor(trainX)
    trainY = torch.Tensor(trainY)
    valX = torch.Tensor(valX)
    valY = torch.Tensor(valY)
    mm=valX

    # 构建Transformer模型
    class TransformerEncoder(nn.Module):
        def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
            super(TransformerEncoder, self).__init__()

            self.mha = nn.MultiheadAttention(embed_dim, num_heads)
            self.layernorm1 = nn.LayerNorm(embed_dim)
            self.dropout1 = nn.Dropout(dropout_rate)

            self.dense1 = nn.Linear(embed_dim, dense_dim)
            self.dense2 = nn.Linear(dense_dim, embed_dim)
            self.layernorm2 = nn.LayerNorm(embed_dim)
            self.dropout2 = nn.Dropout(dropout_rate)

        def forward(self, inputs):
            attn_output, _ = self.mha(inputs, inputs, inputs)
            attn_output = self.dropout1(attn_output)
            out1 = self.layernorm1(inputs + attn_output)

            dense_output = self.dense1(out1)
            dense_output = self.dense2(dense_output)
            dense_output = self.dropout2(dense_output)
            out2 = self.layernorm2(out1 + dense_output)

            return out2


    class TransformerDecoder(nn.Module):
        def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
            super(TransformerDecoder, self).__init__()

            self.mha1 = nn.MultiheadAttention(embed_dim, num_heads)
            self.mha2 = nn.MultiheadAttention(embed_dim, num_heads)
            self.layernorm1 = nn.LayerNorm(embed_dim)
            self.layernorm2 = nn.LayerNorm(embed_dim)
            self.layernorm3 = nn.LayerNorm(embed_dim)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.dropout3 = nn.Dropout(dropout_rate)

            self.dense1 = nn.Linear(embed_dim, dense_dim)
            self.dense2 = nn.Linear(dense_dim, embed_dim)
            self.layernorm4 = nn.LayerNorm(embed_dim)
            self.dropout4 = nn.Dropout(dropout_rate)

        def forward(self, inputs, encoder_outputs):
            attn1, _ = self.mha1(inputs, inputs, inputs)
            attn1 = self.dropout1(attn1)
            out1 = self.layernorm1(inputs + attn1)

            attn2, _ = self.mha2(out1, encoder_outputs, encoder_outputs)
            attn2 = self.dropout2(attn2)
            out2 = self.layernorm2(out1 + attn2)

            dense_output = self.dense1(out2)
            dense_output = self.dense2(dense_output)
            dense_output = self.dropout3(dense_output)
            out3 = self.layernorm3(out2 + dense_output)

            decoder_output = self.dense1(out3)
            decoder_output = self.dense2(decoder_output)
            decoder_output = self.dropout4(decoder_output)
            out4 = self.layernorm4(out3 + decoder_output)

            return out4


    class Transformer(nn.Module):
        def __init__(self, num_features, embed_dim, dense_dim, num_heads, dropout_rate, num_blocks, output_sequence_length):
            super(Transformer, self).__init__()

            self.embedding = nn.Linear(num_features, embed_dim)
            self.transformer_encoder = nn.ModuleList(
                [TransformerEncoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)])
            self.transformer_decoder = nn.ModuleList(
                [TransformerDecoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)])
            self.final_layer = nn.Linear(embed_dim * look_back, output_sequence_length)

        def forward(self, inputs):
            encoder_inputs = inputs
            decoder_inputs = inputs

            encoder_outputs = self.embedding(encoder_inputs)
            for i in range(len(self.transformer_encoder)):
                encoder_outputs = self.transformer_encoder[i](encoder_outputs)

            decoder_outputs = self.embedding(decoder_inputs)
            for i in range(len(self.transformer_decoder)):
                decoder_outputs = self.transformer_decoder[i](decoder_outputs, encoder_outputs)

            decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[1] * decoder_outputs.shape[2])
            decoder_outputs = self.final_layer(decoder_outputs)
            decoder_outputs = decoder_outputs.view(-1, T)
            return decoder_outputs


    # 定义训练集和测试集的数据加载器
    class MyDataset(Dataset):
        def __init__(self, data_X, data_Y):
            self.data_X = data_X
            self.data_Y = data_Y

        def __getitem__(self, index):
            x = self.data_X[index]
            y = self.data_Y[index]
            return x, y

        def __len__(self):
            return len(self.data_X)
    class MyDataset1(Dataset):
        def __init__(self, X):
            self.data_X = X


        def __getitem__(self, index):
            x = self.data_X[index]

            return x

        def __len__(self):
            return len(self.data_X)

    train_dataset = MyDataset(trainX, trainY)
    val_dataset = MyDataset(valX, valY)
    predict_data=MyDataset1(predict_data)
    #print(type(predict_data))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # 创建模型实例
    model = Transformer(num_features=num_features, embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads,
                        dropout_rate=dropout_rate, num_blocks=num_blocks, output_sequence_length=T)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    train_losses = []
    val_losses = []
    best_val_loss=1000000
    # 训练模型
    for epoch in range(epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, position=0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # 在验证集上计算损失
        total_val_loss = 0.0
        num_val_batches = 0
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, position=0):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches
        train_losses.append(loss.item())  # 注意：这里的loss是最后一个训练批次的loss，可能不是整个epoch的平均
        val_losses.append(avg_val_loss)

        # 检查是否是最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model

            # 每个epoch打印一次训练和验证损失
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    model_path='./model_tran/model_tran1' + station + '.pth'
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    torch.save(best_model,model_path)
    #可视化损失函数
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    model= torch.load(model_path)
    model.to(device)

    # 测试模型
    model.eval()
    predictions = []
    trr = np.zeros((75, 5))
    # print(mm)
    # print(trr)
    for z in range(0,5):
            trr[z,:]=mm[-1,z-5,:]

    with torch.no_grad():
        for i in range(24):
            inputs =torch.tensor(trr[i:i+5,:],dtype=torch.float32)
            inputs = inputs.view(1, 5, 5)

            inputs =inputs.to(device)

            outputs = model(inputs)
            outputs=outputs.cpu()

            predictions.append(outputs)

            trr[i+5,:]=np.concatenate((predict_data[i,:],[outputs]))

    predictions_np = torch.stack(predictions)  # 将张量转移到CPU并转换为NumPy数组
    predictions_np=predictions_np.cpu().numpy()
    predictions_reshaped = predictions_np.reshape(-1, 1)  # 重塑数组


    # 测试集数据反归一化
    predictions = scaler2.inverse_transform(predictions_reshaped)

    print(type(predictions))
    predictions=predictions.flatten()
    # 假设 predictions 是一个 NumPy 数组，其中每个元素都是一个 float32 类型的值
    for j,prediction in enumerate(predictions):
        # 如果只写入一个值，使用 write 方法而不是 write_row
        worksheet.write(j+sta*24,1, prediction)  # 注意这里的行索引是固定的 2，列索引 i 会变化
workbook.close()



