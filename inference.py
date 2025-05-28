import matplotlib.pyplot as plt
import torch
from models.sit import SIT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型结构并加载 state_dict
model = SIT().to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))  # 加载权重
model.eval()

# 初始化起始噪声图像
x = torch.randn(size=(1, 1, 28, 28), device=device)
steps = 250
label = 3

with torch.no_grad():
    for i in range(steps):
        t = torch.tensor([1.0 / steps * i], device=device)
        label_tensor = torch.tensor([label], dtype=torch.long, device=device)
        pred_vt = model(x, t, label_tensor)
        x = x + pred_vt * (1.0 / steps)
        x = x.detach()  # 防止梯度累积

# 后处理为图像格式 [0,1]
x = (x + 1) / 2
plt.figure(figsize=(1, 1))
plt.axis('off')
plt.imshow(x[0, 0].cpu().numpy(), cmap='gray')
plt.savefig('result.png')
