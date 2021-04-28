import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 读取训练数据
training_input = pd.read_csv('dataset/training_input.csv')
training_output = pd.read_csv('dataset/training_output.csv')
# 将数组转化为张量
n_train = training_input.shape[0]
training_input_tensor = torch.tensor(training_input[:n_train].values,dtype=torch.float)
training_output_tensor = torch.tensor(training_output[:n_train].values,dtype=torch.float)

# 读取测试数据并合并
test_input = pd.read_csv('dataset/test_input.csv')
test_output = pd.read_csv('dataset/test_output.csv')
# 将数组转化为张量
n_test = test_input.shape[0]
test_input_tensor = torch.tensor(test_input[:n_train].values, dtype=torch.float)
test_output_tensor = torch.tensor(test_output[:n_train].values, dtype=torch.float)


# 定义网络和损失函数
class Net(torch.nn.Module):  # 开始搭建一个神经网络
    def __init__(self, n_feature, n_hidden, n_output):  # 神经网络初始化，设置输入层参数，隐藏层参数，输出层参数
        super(Net, self).__init__()  # 用super函数调用父类的通用初始化函数初始一下
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 设置隐藏层的输入输出参数，比如说输入是n_feature,输出是n_hidden
        self.out = torch.nn.Linear(n_hidden, n_output)  # 同样设置输出层的输入输出参数

    def forward(self, x):  # 前向计算过程
        x = F.relu(self.hidden(x))  # 样本数据经过隐藏层然后被Relu函数掰弯！
        x = self.out(x)
        #经过输出层返回
        return x

net = Net(n_feature=6, n_hidden=10, n_output=3)  # two classification has two n_features#实例化一个网络结构
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 设置优化器参数,lr=0.002指的是学习率的大小
loss_func = torch.nn.BCEWithLogitsLoss()  # 将损失函数设置为loss_func 不适用:CrossEntropyLoss

plt.ion()

# 用训练数据训练网络
for t in range(10000):  # 迭代次数
    out = net(training_input_tensor)
    loss = loss_func(out, training_output_tensor)  # 计算loss为out和y的差异
    print(t, loss)

    optimizer.zero_grad()  # 清除一下上次梯度计算的数值
    loss.backward()  # 进行反向传播
    optimizer.step()  # 最优化迭代

print('Finished Training')

# 用测试数据测试网络
out = net(test_input_tensor)
print(out)
print(test_output_tensor)
loss = loss_func(out, test_output_tensor)
print(loss)

