import numpy as np

# 加载 .npz 文件
data = np.load('Wafer_Map_Datasets.npz')

# 查看文件内容
print("File contains the following arrays:", data.files)

# 读取特定数组
array1 = data['array1']
array2 = data['array2']

# 打印数组内容
print("Array 1:\n", array1)
print("Array 2:\n", array2)

# 关闭文件（可选）
data.close()