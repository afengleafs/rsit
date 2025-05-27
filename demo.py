import torch
from models.utils import apply_rotary_emb, precompute_freqs_cis_2d,RAttention
from timm.models.vision_transformer import PatchEmbed
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

    pos = precompute_freqs_cis_2d(dim // num_heads, H, W)
    r_attn = RAttention(dim, num_heads=num_heads)
    x_out = r_attn(x_out, pos, mask=None)
    print(x_out.shape)


