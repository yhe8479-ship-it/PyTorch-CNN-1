# LeNet模型的搭建
import torch
from torch import nn
# 展示参数的包
from torchsummary import summary

# 1.初始化：定义网络层；激活函数
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.sig = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c3 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(400,120)
        self.f6 = nn.Linear(120,84)
        self.f7 = nn.Linear(84,10)

# 2.搭建LeNet-5网络前向传播过程c1 -> sig -> s2 -> c3 -> sig -> s4 -> flatten -> f5 -> f6 -> f7;
#   把数据x传进去;
    def forward(self,x):
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

if __name__=="__main__":
    # 判断用CPU还是GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 把model放进设备里，即实例化模型
    model = LeNet().to(device)
    print(summary(model,(1,28,28)))



























