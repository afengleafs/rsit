import matplotlib.pyplot as plt
import torch

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=torch.load('model.pt').to(device)

x=torch.randn(size=(1,1,28,28)).to(device)
steps=250
label=5

model.eval()
with torch.no_grad():
    for i in range(steps):
        t=torch.tensor([1.0/steps*i]).to(device)
        label=torch.tensor([label],dtype=torch.long).to(device)
        pred_vt=model(x,t,label)
        x=x+pred_vt*1.0/steps
        x=x.detach()
    
x=(x+1)/2
plt.figure(figsize=(1,1))
plt.axis('off')
plt.imshow(x[0,0].cpu().numpy(),cmap='gray')