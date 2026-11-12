"""E(3)-equivariant noise predictor (EGNN-style) and DDPM schedule/sampler."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .data import ELEMENTS, N_BOND


def sinusoidal(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class EGNNLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(2 * h + 1 + N_BOND + h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.coord = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self.node = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))
        self.norm = nn.LayerNorm(h)

    def forward(self, h, x, bond1h, mask, temb):
        B, N, H = h.shape
        diff = x[:, :, None] - x[:, None]                 # (B,N,N,3)
        d2 = (diff ** 2).sum(-1, keepdim=True)
        hi = h[:, :, None].expand(B, N, N, H)
        hj = h[:, None].expand(B, N, N, H)
        t = temb[:, None, None].expand(B, N, N, H)
        m = self.edge(torch.cat([hi, hj, d2, bond1h, t], -1))
        pair = (mask[:, :, None] & mask[:, None]) & ~torch.eye(N, dtype=torch.bool, device=h.device)[None]
        m = m * pair[..., None]
        w = self.coord(m) * pair[..., None]
        dx = (diff * w).sum(2) / mask.sum(1).clamp(min=1)[:, None, None]
        h = self.norm(h + self.node(torch.cat([h, m.sum(2)], -1)))
        return h * mask[..., None], dx


class Denoiser(nn.Module):
    """Predicts the noise eps added to centred coordinates, as an equivariant vector field."""

    def __init__(self, hidden=64, layers=4):
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(len(ELEMENTS) + 1, hidden)
        self.tmlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(EGNNLayer(hidden) for _ in range(layers))

    def forward(self, x, t, z, bond, mask):
        temb = self.tmlp(sinusoidal(t, self.hidden))
        h = self.emb(z)
        bond1h = nn.functional.one_hot(bond, N_BOND).float()
        out = torch.zeros_like(x)
        for layer in self.layers:
            h, dx = layer(h, x, bond1h, mask, temb)
            out = out + dx
        out = out * mask[..., None]
        out = out - out.sum(1, keepdim=True) / mask.sum(1).clamp(min=1)[:, None, None]
        return out * mask[..., None]  # re-mask: mean subtraction would otherwise leak into padding


class Diffusion:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.05, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)

    @staticmethod
    def centre(x, mask):
        mu = (x * mask[..., None]).sum(1, keepdim=True) / mask.sum(1).clamp(min=1)[:, None, None]
        return (x - mu) * mask[..., None]

    def noise(self, x0, t, mask):
        eps = self.centre(torch.randn_like(x0), mask)
        a = self.abar[t][:, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * eps, eps

    def loss(self, model, batch):
        x0, mask = batch["x"], batch["mask"]
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        xt, eps = self.noise(x0, t, mask)
        pred = model(xt, t, batch["z"], batch["bond"], mask)
        sq = ((pred - eps) ** 2) * mask[..., None]  # padded atoms contribute nothing
        return (sq.sum((1, 2)) / (mask.sum(1) * 3)).mean()

    @torch.no_grad()
    def sample(self, model, batch):
        mask = batch["mask"]
        x = self.centre(torch.randn_like(batch["x"]), mask)
        for i in reversed(range(self.T)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            eps = model(x, t, batch["z"], batch["bond"], mask)
            a, ab, b = self.alphas[i], self.abar[i], self.betas[i]
            x = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            if i > 0:
                x = x + b.sqrt() * self.centre(torch.randn_like(x), mask)
            x = self.centre(x, mask)
        return x
