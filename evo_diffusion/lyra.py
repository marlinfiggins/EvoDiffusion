import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)

class PGC(nn.Module):
    def __init__(self, d_model, expansion_factor=2.0, dropout=0.0):
        """
        Parallel Gated Convolution (PGC) block.

        Args:
            d_model (int): Input/output dimension.
            expansion_factor (float): Factor to expand hidden dim before gating.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        hidden_dim = int(d_model * expansion_factor)

        self.in_proj = nn.Linear(d_model, hidden_dim * 2)
        self.in_norm = RMSNorm(hidden_dim * 2)

        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)

        self.norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        """
        Args:
            u: Tensor of shape (B, L, D)
        Returns:
            Tensor of shape (B, L, D)
        """
        x = self.in_proj(u)                              # (B, L, 2H)
        x = self.in_norm(x)
        x, v = x.chunk(2, dim=-1)                        # (B, L, H), (B, L, H)

        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)  # Depthwise conv (B, L, H)
        gate = v * x_conv
        out = self.out_proj(self.norm(gate))             # (B, L, D)
        return self.dropout(out)

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        self.p = p
        self.tie = tie
        self.transposed = transposed
        if not (0 <= self.p < 1.0):
            raise ValueError("dropout probability must be in [0, 1)")

    def forward(self, X):
        if self.training and self.p > 0:
            if not self.transposed:
                # Transpose to (B, D, ...)
                dims = list(range(X.ndim))
                X = X.permute(0, -1, *dims[1:-1])
            shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(shape, device=X.device) < (1. - self.p)
            X = X * mask / (1.0 - self.p)
            if not self.transposed:
                # Transpose back to original layout
                dims = list(range(X.ndim))
                X = X.permute(0, *dims[2:], 1)
        return X


class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.register("log_dt", log_dt, lr)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))  # shape (H, N//2, 2)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        self.register("log_A_real", log_A_real, lr)

        A_imag = math.pi * torch.arange(N // 2).unsqueeze(0).expand(H, -1)
        self.register("A_imag", A_imag, lr)

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
        optim = {"weight_decay": 0.0}
        if lr is not None:
            optim["lr"] = lr
        getattr(self, name)._optim = optim

    def forward(self, L):
        dt = torch.exp(self.log_dt)  # (H,)
        C = torch.view_as_complex(self.C)  # (H, N//2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N//2)
        dtA = A * dt.unsqueeze(-1)  # (H, N//2)
        t = torch.arange(L, device=A.device)
        K = dtA.unsqueeze(-1) * t  # (H, N//2, L)
        C = C * (torch.exp(dtA) - 1.) / A  # (H, N//2)
        kernel = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real  # (H, L)
        return kernel


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout, transposed=transposed) if dropout > 0.0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=1)  # apply gating along channel dim
        )

    def forward(self, u):
        """
        Args:
            u: (B, H, L) if transposed=True else (B, L, H)
        """
        if not self.transposed:
            u = u.transpose(-1, -2)  # (B, H, L)

        L = u.size(-1)
        k = self.kernel(L=L)  # (H, L)

        # Frequency-domain convolution
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]

        # Add skip connection
        y = y + u * self.D.unsqueeze(-1)

        # Post-processing
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        if not self.transposed:
            y = y.transpose(-1, -2)  # (B, L, H)
        return y

class S4DEmbedding(nn.Module):
    def __init__(self, sequence_length, input_dim, embedding_dim, num_layers=1, d_state=64, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        self.pos_embedding = nn.Embedding(sequence_length, embedding_dim)
        self.layers = nn.ModuleList([
            S4D(d_model=embedding_dim, d_state=d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.input_proj(x)  # (B, L, D)
        B, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_embedding(positions)

        x = x.transpose(1, 2)  # (B, D, L)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)  # (B, L, D)
        return self.norm(x)


# --- PGCEmbedding ---
class PGCEmbedding(nn.Module):
    def __init__(self, sequence_length, input_dim, embedding_dim, num_layers=1, expansion_factor=2.0, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        self.pos_embedding = nn.Embedding(sequence_length, embedding_dim)
        self.layers = nn.ModuleList([
            PGC(d_model=embedding_dim, expansion_factor=expansion_factor, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.input_proj(x)
        B, L, _ = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Lyra(nn.Module):
    def __init__(
        self,
        sequence_length,
        input_dim,
        embedding_dim,
        expansion_factor=2.0,
        s4_state_dim=64,
        dropout=0.0,
        use_residual=True,
        num_layers=1
    ):
        """
        Lyra block combining PGC and S4D.

        Args:
            sequence_length (int): Length of input sequence.
            input_dim (int): Input feature dimension (e.g. 4 for one-hot DNA).
            embedding_dim (int): Output embedding dimension.
            expansion_factor (float): Expansion factor for PGC hidden dim.
            s4_state_dim (int): Internal state dimension for S4D.
            dropout (float): Dropout rate.
            use_residual (bool): Whether to apply residual connections.
            num_layers (int): Number of stacked Lyra layers.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        self.pos_embedding = nn.Embedding(sequence_length, embedding_dim)

        self.layers = nn.ModuleList([
            LyraBlock(
                embedding_dim=embedding_dim,
                expansion_factor=expansion_factor,
                s4_state_dim=s4_state_dim,
                dropout=dropout,
                use_residual=use_residual
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (B, L, input_dim)
        Returns:
            (B, L, embedding_dim)
        """
        x = self.input_proj(x)
        B, L, _ = x.shape
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_embedding(pos_ids)

        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class LyraBlock(nn.Module):
    def __init__(self, embedding_dim, expansion_factor, s4_state_dim, dropout, use_residual):
        super().__init__()
        self.pgc = PGC(d_model=embedding_dim, expansion_factor=expansion_factor, dropout=dropout)
        self.s4d = S4D(d_model=embedding_dim, d_state=s4_state_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.use_residual = use_residual

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        residual = x
        x = self.pgc(self.norm1(x))
        if self.use_residual:
            x = x + residual

        residual = x
        x = self.s4d(self.norm2(x))
        if self.use_residual:
            x = x + residual

        return x
