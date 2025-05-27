import torch.nn as nn
from .utils import RMSNorm, RAttention, FeedForward,modulate,precompute_freqs_cis_2d
from timm.models.vision_transformer import PatchEmbed
import torch

class SiTBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4.0, ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = RAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x,  c, pos, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos, mask=mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    

if __name__ == "__main__":
    batch_size = 2
    dim = 128
    path_size = 4
    height = 16
    width = 16
    H = height // path_size
    W = width // path_size
    channel = 3
    num_heads = 4
    
    x = torch.randn(batch_size, channel, height, width)
    pathch_emd = PatchEmbed(
        img_size=(height, width),
        patch_size=path_size,
        in_chans=channel,
        embed_dim=dim,
    )
    x_out = pathch_emd(x)
    print(x_out.shape)
    c = torch.randn(batch_size, path_size*path_size,dim)

    pos = precompute_freqs_cis_2d(dim // num_heads, H, W)
    r_attn = RAttention(dim, num_heads=num_heads)
    s_out = r_attn(x_out, pos, mask=None)
    print(s_out.shape)