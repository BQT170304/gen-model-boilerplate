import torch
import torch.nn as nn
import torch.nn.functional as F

#Follow https://github.com/w86763777/pytorch-ddpm/blob/master/model.py
class AttnBlock(nn.Module):
    """
    ## Attention block
    """

    def __init__(self, channels, n_heads=None, n_layers=None, d_cond=None):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.proj_q = nn.Conv2d(channels, channels, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(channels, channels, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(channels, channels, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(channels, channels, 1, stride=1, padding=0)
        self.initialize()

    def forward(self, x, cond=None):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C)**(-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        h = self.proj(h)

        return x + h