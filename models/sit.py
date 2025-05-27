import torch
import torch.nn as nn
from .sit_block import SiTBlock
from .utils import precompute_freqs_cis_2d,TimestepEmbedder,LabelEmbedder,FinalLayer,Embed

device = "cuda" if torch.cuda.is_available() else "cpu"

class SIT(nn.Module):
    def __init__(
            self,
            in_channels=1,
            num_groups=4,
            hidden_size=128,
            num_blocks=8,
            patch_size=2,
            num_classes=10,
            learn_sigma=True,
            deep_supervision=0,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.x_embedder = Embed(in_channels*patch_size**2, hidden_size, bias=True)
        self.s_embedder = Embed(in_channels*patch_size**2, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes+1, hidden_size,dropout_prob=0.0)

        self.final_layer = FinalLayer(hidden_size, in_channels*patch_size**2)

        self.weight_path = weight_path

        self.load_ema = load_ema
        self.blocks = nn.ModuleList([
            SiTBlock(self.hidden_size, self.num_groups) for _ in range(self.num_blocks)
        ])
        self.initialize_weights()
        self.precompute_pos = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, y,mask=None):
        B, _, H, W = x.shape
        pos = self.fetch_pos(H//self.patch_size, W//self.patch_size, x.device)
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y,train=True).view(B, 1, self.hidden_size)
        c = nn.functional.silu(t + y)
        x = self.x_embedder(x)
        for i in range(self.num_blocks):
            x = self.blocks[i](x, c, pos, None)
        x = self.final_layer(x, c)
        v = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return v
    

if __name__ == "__main__":
    B, C, H, W      = 4, 3, 32, 32
    patch_size      = 2
    num_classes     = 10
    t_max           = 1000

    model = SIT(
        in_channels=C,
        hidden_size=128,
        num_groups=4,
        num_blocks=8,
        patch_size=patch_size,
        num_classes=num_classes,
    ).to(device)

    x = torch.randn(B, C, H, W, device=device, requires_grad=True)
    t = torch.randint(0, t_max, (B,), device=device)
    y = torch.randint(0, num_classes, (B,), device=device)

    v = model(x, t, y) 

    print(x.shape)
    print(v.shape)              



