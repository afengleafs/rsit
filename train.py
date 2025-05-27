from models.sit import SIT
from dataset_mnist.dataset import MNIST
import torch
import os


model = SIT()
dataset = MNIST()
dataloader=torch.utils.data.DataLoader(dataset,batch_size=64,shuffle=True)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.train()


log_interval = 100          # 每隔多少个 batch 打印一次损失
save_path    = "model.pt"   # 最终模型文件名

if __name__ == "__main__":
    for step, (batch_img, batch_labels) in enumerate(dataloader, start=1):
        batch_img    = batch_img.to(device)
        batch_labels = batch_labels.to(device)

        # 随机时间步、噪声与合成输入
        batch_t    = torch.rand(batch_img.size(0), device=device)
        batch_noise = torch.randn_like(batch_img)
        batch_xt    = (1.0 - batch_t.view(-1, 1, 1, 1)) * batch_noise + \
                      batch_t.view(-1, 1, 1, 1) * batch_img

        # 前向 + 反向
        pred_vt = model(batch_xt, batch_t, batch_labels)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(pred_vt, batch_img - batch_noise)
        loss.backward()
        optimizer.step()

        # 按指定间隔输出当前损失
        if step % log_interval == 0:
            print(f"[step {step:>6}] loss = {loss.item():.6f}")

    # 训练结束后保存模型
    torch.save(model.state_dict(), ".model.pt")
    os.replace(".model.pt", save_path)
    print(f"Training finished. Final loss = {loss.item():.6f} | Model saved to {save_path}")
