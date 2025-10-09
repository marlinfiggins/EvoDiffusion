import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        \"\"\"Sinusoidal time embedding.
        Args:
            t: [B] or [B,1] float times
        Returns:
            emb: [B, dim]
        \"\"\"
        t = t.reshape(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(0, half, device=t.device) / max(1, half))
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_ff: int = 1024,
        max_len: int = 512,
        pad_id: int = 0,
        cls_token: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.cls_token = cls_token
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len + (1 if cls_token else 0), d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        if cls_token:
            self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            self.register_parameter("cls", None)

        self.pool = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Encode tokens.
        Args:
            x: [B, L] token ids
        Returns:
            H: [B, L', d] token representations
            s: [B, d] pooled representation
        \"\"\"
        B, L = x.shape
        device = x.device

        if self.cls_token:
            cls_tok = self.cls.expand(B, 1, -1)
            tok = self.token_emb(x)
            seq = torch.cat([cls_tok, tok], dim=1)
            pos_idx = torch.arange(0, L + 1, device=device).unsqueeze(0).repeat(B, 1)
        else:
            seq = self.token_emb(x)
            pos_idx = torch.arange(0, L, device=device).unsqueeze(0).repeat(B, 1)

        seq = seq + self.pos_emb(pos_idx)
        pad_mask = (x == self.pad_id)
        if self.cls_token:
            src_kpm = F.pad(pad_mask, (1, 0), value=False)
        else:
            src_kpm = pad_mask

        H = self.encoder(seq, src_key_padding_mask=src_kpm)
        if self.cls_token:
            s = H[:, 0]
            tokens = H[:, 1:]
        else:
            s = H.mean(dim=1)
            tokens = H

        s = self.pool(s)
        return tokens, s


class ProjectionToLatent(nn.Module):
    def __init__(self, d_in: int, d_latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_latent),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, d_model)
        self.to_beta = nn.Linear(cond_dim, d_model)
        nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight); nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond).unsqueeze(1)
        beta = self.to_beta(cond).unsqueeze(1)
        h = F.layer_norm(h, h.shape[-1:])
        return (1.0 + gamma) * h + beta


class MaskedPredictor(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        cond_dim: int,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_ff: int = 1024,
    ):
        super().__init__()
        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.ModuleDict({
                "film": FiLM(cond_dim, d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ff": nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, dim_ff),
                    nn.GELU(),
                    nn.Linear(dim_ff, d_model),
                )
            }))
        self.blocks = nn.ModuleList(blocks)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, H_tokens: torch.Tensor, cond: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = H_tokens
        for blk in self.blocks:
            h = blk["film"](h, cond)
            h_resid = h
            attn_out, _ = blk["attn"](h, h, h, need_weights=False, attn_mask=attn_mask)
            h = h_resid + attn_out
            h = h + blk["ff"](h)
        h = self.ln_out(h)
        logits = self.head(h)
        return logits


@dataclass
class DynamicsConfig:
    process: str = "brownian"  # "brownian" or "ou"
    sigma: float = 1.0
    kappa: float = 0.2


def brownian_inc_nll(z_s: torch.Tensor, z_t: torch.Tensor, dt: torch.Tensor, sigma: float) -> torch.Tensor:
    \"\"\"Per-sample Brownian increment NLL.\"\"\"
    d = z_s.shape[-1]
    diff = z_t - z_s
    var = (sigma ** 2) * dt.unsqueeze(-1)
    term = (diff * diff / (2.0 * var)).sum(-1) + 0.5 * d * torch.log(var.squeeze(-1) + 1e-8)
    return term


def ou_inc_nll(z_s: torch.Tensor, z_t: torch.Tensor, dt: torch.Tensor, sigma: float, kappa: float) -> torch.Tensor:
    \"\"\"Per-sample OU increment NLL.\"\"\"
    d = z_s.shape[-1]
    alpha = torch.exp(-kappa * dt).unsqueeze(-1)
    mean = alpha * z_s
    var = (sigma ** 2) / (2.0 * kappa) * (1.0 - torch.exp(-2.0 * kappa * dt)).unsqueeze(-1)
    diff = z_t - mean
    term = (diff * diff / (2.0 * var)).sum(-1) + 0.5 * d * torch.log(var.squeeze(-1) + 1e-8)
    return term


def brownian_bridge_nll(z0: torch.Tensor, zu: torch.Tensor, zT: torch.Tensor, u: torch.Tensor, T: torch.Tensor, sigma: float) -> torch.Tensor:
    \"\"\"Per-sample Brownian bridge NLL for intermediate time u in (0,T).\"\"\"
    d = z0.shape[-1]
    alpha = (u / T).unsqueeze(-1)
    mean = (1.0 - alpha) * z0 + alpha * zT
    var = (sigma ** 2) * (u * (T - u) / T).unsqueeze(-1)
    diff = zu - mean
    term = (diff * diff / (2.0 * var)).sum(-1) + 0.5 * d * torch.log(var.squeeze(-1) + 1e-8)
    return term


def ou_posterior_z0_given_zt(z_t: torch.Tensor, t: torch.Tensor, sigma: float, kappa: float) -> Tuple[torch.Tensor, torch.Tensor]:
    \"\"\"Posterior for OU with stationary prior: returns (mean, std).\"\"\"
    Sigma_star = (sigma ** 2) / (2.0 * kappa)
    alpha = torch.exp(-kappa * t).unsqueeze(-1)
    mean = alpha * z_t
    std = torch.sqrt(torch.clamp(Sigma_star * (1.0 - torch.exp(-2.0 * kappa * t)), min=1e-8)).unsqueeze(-1)
    return mean, std


def brownian_posterior_z0_given_zt(z_t: torch.Tensor, t: torch.Tensor, mu0: torch.Tensor, Sigma0: float, sigma: float) -> Tuple[torch.Tensor, torch.Tensor]:
    \"\"\"Posterior for Brownian with Gaussian root prior: returns (mean, std).\"\"\"
    inv_S0 = 1.0 / Sigma0
    inv_St = 1.0 / (sigma ** 2 * torch.clamp(t, min=1e-8))
    Sigma_post = 1.0 / (inv_S0 + inv_St)
    mu_post = Sigma_post.unsqueeze(-1) * (inv_S0 * mu0 + inv_St.unsqueeze(-1) * z_t)
    std = torch.sqrt(torch.clamp(Sigma_post, min=1e-8)).unsqueeze(-1)
    return mu_post, std
