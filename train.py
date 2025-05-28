import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from models.sit import SIT
from dataset_mnist.dataset import MNIST
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SIT().to(device)
dataset = MNIST(train=True, to_minus1_1=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
num_epochs = 10
log_interval = 100
save_path = "model.pt"

if __name__ == "__main__":  # 修复语法错误
    for epoch in range(num_epochs):  # 添加epoch循环
        epoch_loss = 0.0
        for step, (batch_img, batch_labels) in enumerate(dataloader, start=1):
            batch_img = batch_img.to(device)
            batch_labels = batch_labels.to(device)
            
            # 随机时间步、噪声与合成输入
            batch_t = torch.rand(batch_img.size(0), device=device)
            batch_noise = torch.randn_like(batch_img, device=device)
            batch_xt = (1.0 - batch_t.view(-1, 1, 1, 1)) * batch_noise + \
                       batch_t.view(-1, 1, 1, 1) * batch_img
            
            # 前向 + 反向
            pred_vt = model(batch_xt, batch_t, batch_labels)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(pred_vt, batch_img - batch_noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 按指定间隔输出当前损失
            if step % log_interval == 0:
                print(f"[Epoch {epoch+1:>2}/{num_epochs}] [Step {step:>6}] loss = {loss.item():.6f}")
        
        print(f"Epoch {epoch+1}/{num_epochs} finished. Average loss = {epoch_loss/len(dataloader):.6f}")
    
    # 训练结束后保存模型
    torch.save(model.state_dict(), ".model.pt")
    os.replace(".model.pt", save_path)
    print(f"Training finished. Model saved to {save_path}")
