"""Paper-informed PyTorch reproduction of IJCAI-24 MetaISP.

No official IJCAI-24 code or checkpoint was found during the project audit. This
module implements the published MCCB, MSCB, MFEB, MRB and wavelet U-Net design;
it must not be described as the authors' original checkpoint or exact source.
Input is packed RGGB NCHW at half sensor resolution; output is rendered RGB at
full sensor resolution.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-6) * self.weight + self.bias


class Predict(nn.Module):
    """Depthwise-convolution + GAP predictor shown in the paper's block key."""
    def __init__(self, channels: int, outputs: int):
        super().__init__()
        hidden = max(8, channels // 4)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, outputs, 1)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.dw(x), 1)
        return self.fc2(F.relu(self.fc1(x), inplace=False)).flatten(1).softmax(dim=1)


class MCCB(nn.Module):
    """Meta channel correction using a weighted set of basis matrices."""
    def __init__(self, channels: int, basis_count: int = 10):
        super().__init__()
        hidden = channels * 2  # K=2C in the paper's correction projection.
        self.norm = LayerNorm2d(channels)
        self.context = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
        self.project_in = nn.Conv2d(channels, hidden, 1)
        self.predict = Predict(channels, basis_count)
        self.bases = nn.Parameter(torch.empty(basis_count, hidden, hidden))
        nn.init.orthogonal_(self.bases)
        self.project_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        y = self.project_in(self.context(self.norm(x)))
        weights = self.predict(x)
        matrix = torch.einsum('bn,nij->bij', weights, self.bases)
        tokens = y.flatten(2).transpose(1, 2)
        y = torch.bmm(tokens, matrix).transpose(1, 2).reshape_as(y)
        return x + self.project_out(y)


class MSCB(nn.Module):
    """Meta spatial correction using cosine similarity to a learned basis vector."""
    def __init__(self, channels: int, basis_count: int = 10):
        super().__init__()
        hidden = channels * 2  # K=2C in the paper's correction projection.
        self.norm = LayerNorm2d(channels)
        self.context = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
        self.project_in = nn.Conv2d(channels, hidden, 1)
        self.predict = Predict(channels, basis_count)
        self.bases = nn.Parameter(torch.empty(basis_count, hidden))
        nn.init.normal_(self.bases, std=hidden ** -0.5)
        self.project_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        y = self.project_in(self.context(self.norm(x)))
        vector = torch.einsum('bn,nc->bc', self.predict(x), self.bases)
        cosine = F.cosine_similarity(y, vector[:, :, None, None], dim=1, eps=1e-6)
        attention = ((cosine + 1) * 0.5).unsqueeze(1)
        return x + self.project_out(y * attention)


class MetaCorrection(nn.Module):
    def __init__(self, channels: int, basis_count: int):
        super().__init__()
        self.channel = MCCB(channels, basis_count)
        self.spatial = MSCB(channels, basis_count)

    def forward(self, x):
        return self.spatial(self.channel(x))


class MFEB(nn.Module):
    """Six adaptively mixed shallow feature branches from the paper."""
    def __init__(self, channels: int, basis_count: int = 10):
        super().__init__()
        self.learned = nn.Conv2d(4, channels, 3, padding=1)
        kernels = torch.tensor([
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ], dtype=torch.float32)
        kernels[0] /= 9
        kernels[1] /= 16
        kernels[2:4] /= 8
        self.register_buffer('fixed_kernels', kernels[:, None, None].repeat(1, 4, 1, 1, 1), persistent=True)
        self.fixed_project = nn.ModuleList(nn.Conv2d(4, channels, 1) for _ in range(5))
        self.predict = Predict(4, 6)

    def forward(self, x):
        branches = [self.learned(x)]
        for i, projector in enumerate(self.fixed_project):
            # Each fixed kernel is applied per RGGB plane before learned channel mixing.
            filtered = F.conv2d(x, self.fixed_kernels[i], padding=1, groups=4)
            branches.append(projector(filtered))
        weights = self.predict(x)
        output = branches[0] * weights[:, 0].view(-1, 1, 1, 1)
        for i, branch in enumerate(branches[1:]):
            output = output + branch * weights[:, i + 1].view(-1, 1, 1, 1)
        return output


class HaarDown(nn.Module):
    """Haar-wavelet analysis followed by learned channel projection."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        base = torch.tensor([
            [[1, 1], [1, 1]],
            [[1, -1], [1, -1]],
            [[1, 1], [-1, -1]],
            [[1, -1], [-1, 1]],
        ], dtype=torch.float32).unsqueeze(1) * 0.5
        self.in_channels = in_channels
        self.register_buffer('haar_filters', base.repeat(in_channels, 1, 1, 1), persistent=True)
        self.project = nn.Conv2d(in_channels * 4, out_channels, 3, padding=1)

    def forward(self, x):
        wavelet = F.conv2d(x, self.haar_filters, stride=2, groups=self.in_channels)
        return self.project(wavelet)


class MetaISP2024(nn.Module):
    """Compact four-level MetaISP reproduction, N=10 basis elements."""
    def __init__(self, width: int = 32, basis_count: int = 10):
        super().__init__()
        # IJCAI-24 specifies widths {C, C, C, 2C} from the top to bottom
        # levels. The bottom feature is at 1/16 the full sensor resolution.
        c0, c1, c2, c3 = width, width, width, width * 2
        self.mfeb = MFEB(c0, basis_count)
        self.enc0 = MetaCorrection(c0, basis_count)
        self.down1 = HaarDown(c0, c1)
        self.enc1 = MetaCorrection(c1, basis_count)
        self.down2 = HaarDown(c1, c2)
        self.enc2 = MetaCorrection(c2, basis_count)
        self.down3 = HaarDown(c2, c3)
        self.enc3 = MetaCorrection(c3, basis_count)
        self.up2 = nn.Conv2d(c3, c2 * 4, 3, padding=1)
        self.dec2 = MetaCorrection(c2, basis_count)
        self.up1 = nn.Conv2d(c2, c1 * 4, 3, padding=1)
        self.dec1 = MetaCorrection(c1, basis_count)
        self.up0 = nn.Conv2d(c1, c0 * 4, 3, padding=1)
        self.dec0 = MetaCorrection(c0, basis_count)
        self.reconstruct = nn.Sequential(nn.Conv2d(c0, c0, 3, padding=1), nn.GELU(), nn.Conv2d(c0, 12, 3, padding=1))
        self.channel_guide = nn.Sequential(nn.Conv2d(c3, c3, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Conv2d(c3, 3, 1))

    def forward(self, x):
        # x is N,4,H/2,W/2. Three wavelet downsamples require /8 alignment
        # at packed resolution, corresponding to /16 at sensor resolution.
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % 8, (-w) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        f0 = self.enc0(self.mfeb(x))
        f1 = self.enc1(self.down1(f0))
        f2 = self.enc2(self.down2(f1))
        f3 = self.enc3(self.down3(f2))
        d2 = self.dec2(F.pixel_shuffle(self.up2(f3), 2) + f2)
        d1 = self.dec1(F.pixel_shuffle(self.up1(d2), 2) + f1)
        d0 = self.dec0(F.pixel_shuffle(self.up0(d1), 2) + f0)
        rgb = F.pixel_shuffle(self.reconstruct(d0), 2)
        rgb = rgb * (0.5 + self.channel_guide(f3).sigmoid())
        return rgb[:, :, :h * 2, :w * 2]


if __name__ == '__main__':
    model = MetaISP2024()
    x = torch.rand(1, 4, 128, 128)
    y = model(x)
    print({'params': sum(p.numel() for p in model.parameters()), 'input': list(x.shape), 'output': list(y.shape)})
