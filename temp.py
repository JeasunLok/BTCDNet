import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# 指定 .mat 文件的路径
file_path = r'data\China_Change_Dataset.mat'

# 加载 .mat 文件
mat_data = scipy.io.loadmat(file_path)

# 输出 .mat 文件的结构和每个数组的详细信息
print("Keys in the .mat file:")
for key in mat_data.keys():
    if not key.startswith('__'):  # 跳过元数据键
        print(f"{key}: {type(mat_data[key])}, shape: {mat_data[key].shape}, dtype: {mat_data[key].dtype}")

# 获取 Multiple 数组
multiple_array = mat_data['Multiple']

# 可视化 Multiple 数组
plt.imshow(multiple_array)
plt.title('Image from Multiple Array')
plt.axis('off')  # 不显示坐标轴
plt.show()

# 获取 Binary 数组
binary_array = mat_data['Binary']

# 可视化 Binary 数组
plt.imshow(binary_array, cmap='gray')  # 使用灰度图显示
plt.title('Binary Array Visualization')
plt.axis('off')  # 不显示坐标轴
plt.show()

data_t1 = mat_data['T1']
data_t2 = mat_data['T2']
data_label = mat_data['Binary']
negative_position = np.array(np.where(data_label==0)).transpose(1,0)
positive_position = np.array(np.where(data_label==1)).transpose(1,0)
data_label[data_label==0]=2 # 1-positive 2-negative
print(negative_position.shape,positive_position.shape,data_label.shape)


selected_negative = np.random.choice(negative_position.shape[0], int(200), replace = False)
selected_positive = np.random.choice(positive_position.shape[0], int(200), replace = False)
selected_negative_position=negative_position[selected_negative]
selected_positive_position=positive_position[selected_positive]
train_label = np.zeros(data_label.shape)
for i in range(0, 200):
    train_label[selected_positive_position[i][0],selected_positive_position[i][1]]=1 # 1-postive
    train_label[selected_negative_position[i][0],selected_negative_position[i][1]]=2 # 2-negative
#--------------测试样本-----------------
test_label=data_label-train_label # remove train_label and other test_label

# 查看 TR 和 TE 的形状
print("train shape:", train_label.shape)
print("test shape:", test_label.shape)

# 测试样本
test_label = data_label - train_label

# 可视化 TR 和 TE
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # 两个子图，TR在左侧
plt.imshow(train_label, cmap='gray')  # 使用灰度图显示
plt.title('Training Labels (TR)')
plt.axis('off')  # 不显示坐标轴

# 可视化 TE
plt.subplot(1, 2, 2)  # TE在右侧
plt.imshow(test_label, cmap='gray')  # 使用灰度图显示
plt.title('Test Samples (TE)')
plt.axis('off')  # 不显示坐标轴

plt.tight_layout()
plt.show()
