import torch
import torch.nn as nn
import thop
from fvcore.nn import FlopCountAnalysis

# 定义一个简单模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc = nn.Linear(36864, 10)

    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        for _ in range(100):
            x = x + torch.rand_like(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 实例化模型和输入
model = MyModel()
input_tensor = torch.randn(1, 1, 28, 28)  # 输入大小：batch_size=1, channels=1, height=28, width=28

# 计算 FLOP 和参数
flops, params = thop.profile(model, inputs=(input_tensor,), verbose=True)

print(f"Total FLOPs: {flops}")
print(f"Total Parameters: {params}")
