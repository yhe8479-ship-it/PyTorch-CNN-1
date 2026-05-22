import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

# 1.初始化:定义激活函数；网络层
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.ReLU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4)
        self.s2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.c3 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.c5 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.c6 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1)
        self.c7 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1)
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(6*6*256,4096)
        self.f2 = nn.Linear(4096,4096)
        self.f3 = nn.Linear(4096,10)

# 2.搭建AlexNet网络前向传播过程c1 -> ReLU -> s2 -> c3 -> ReLU -> s4 ->
    #                      c5 -> ReLU ->c6 -> ReLU -> c7 -> ReLU -> s8 -> flatten ->
    #                      f1 -> ReLU -> Dropout -> f2 -> ReLU -> Dropout -> f3
#   把数据x传进去;
    def forward(self,x):
        x = self.ReLU(self.c1(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s4(x)
        x = self.ReLU(self.c5(x))
        x = self.ReLU(self.c6(x))
        x = self.ReLU(self.c7(x))
        x = self.s8(x)

        x = self.flatten(x)
        x = self.ReLU(self.f1(x))
        x = F.dropout(x,0.5)
        x = self.ReLU(self.f2(x))
        x = F.dropout(x, 0.5)
        x = self.f3(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 把model放进设备里，即实例化模型
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))