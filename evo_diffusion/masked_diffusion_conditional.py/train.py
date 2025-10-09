import argparse

import numpy as np
import torch
import torch.optim as optim
from losses import bridge_nll, increment_nll, masked_cross_entropy
from model import (
    DynamicsConfig,
    MaskedPredictor,
    ProjectionToLatent,
    SinusoidalTimeEmbedding,
    TransformerEncoder,
)


class ToySeqDataset(torch.utils.data.Dataset):
    def __init__(self, n: int, L: int, vocab_size: int, pad_id: int = 0):
        super().__init__()
        self.n = n
        self.L = L
        self.vocab = vocab_size
        self.pad_id = pad_id
        rng = np.random.default_rng(42)
        self.base = rng.integers(low=1, high=vocab_size, size=(n, L), dtype=np.int64)
        self.times = rng.uniform(low=0.0, high=5.0, size=(n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.from_numpy(self.base[idx].copy())
        t = float(self.times[idx])
        return x, t


def collate_pair(batch, pad_id: int, mask_token: int, mask_prob: float = 0.25):
    xs, ts = zip(*batch)
    X = torch.stack(xs, dim=0)
    T = torch.tensor(ts, dtype=torch.float32)
    B, L = X.shape
    idx = torch.randperm(B)
    half = B // 2
    s_idx = idx[:half]
    t_idx = idx[half : 2 * half]

    x_s = X[s_idx]
    x_t = X[t_idx]
    t_s = T[s_idx]
    t_t = T[t_idx]

    swap = t_s >= t_t
    tmp = t_s[swap].clone()
    t_s[swap] = t_t[swap]
    t_t[swap] = tmp

    dt = t_t - t_s

    mask = torch.rand_like(x_t, dtype=torch.float32) < mask_prob
    x_t_masked = x_t.clone()
    x_t_masked[mask] = mask_token

    return dict(
        x_s=x_s, t_s=t_s, x_t=x_t, t_t=t_t, dt=dt, x_t_masked=x_t_masked, mask_idx=mask
    )


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print("Device:", device)

    vocab_size = args.vocab
    pad_id = 0
    mask_token = vocab_size - 1

    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        max_len=args.seq_len,
        pad_id=pad_id,
        cls_token=True,
    ).to(device)

    proj = ProjectionToLatent(d_in=args.d_model, d_latent=args.d_latent).to(device)
    time_emb = SinusoidalTimeEmbedding(args.t_dim).to(device)
    cond_dim = args.t_dim + args.d_latent
    predictor = MaskedPredictor(
        d_model=args.d_model,
        vocab_size=vocab_size,
        cond_dim=cond_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
    ).to(device)

    dyn = DynamicsConfig(process=args.process, sigma=args.sigma, kappa=args.kappa)

    params = (
        list(encoder.parameters())
        + list(proj.parameters())
        + list(predictor.parameters())
    )
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    if args.toy:
        ds = ToySeqDataset(
            n=args.n_data, L=args.seq_len, vocab_size=vocab_size, pad_id=pad_id
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_pair(b, pad_id, mask_token, args.mask_prob),
        )
    else:
        raise NotImplementedError(
            "Provide your own dataset with (x_s, t_s), (x_t, t_t), dt, and masked x_t."
        )

    encoder.train()
    proj.train()
    predictor.train()

    for epoch in range(1, args.epochs + 1):
        losses = {"mask": 0.0, "inc": 0.0, "bridge": 0.0}
        n_steps = 0
        for batch in dl:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            x_s, t_s = batch["x_s"], batch["t_s"]
            x_t, t_t = batch["x_t"], batch["t_t"]
            dt = batch["dt"]
            x_t_masked = batch["x_t_masked"]
            mask_idx = batch["mask_idx"]

            H_s, s_s = encoder(x_s)
            H_t, s_t = encoder(x_t_masked)

            z_s = proj(s_s)
            z_t = proj(s_t)

            L_inc = increment_nll(z_s, z_t, dt, dyn)
            if args.use_bridge:
                u = (t_s + t_t) / 2.0
                alpha = (u - t_s) / (t_t - t_s + 1e-8)
                z_u = (1 - alpha.unsqueeze(-1)) * z_s + alpha.unsqueeze(-1) * z_t
                L_bridge = bridge_nll(z_s, z_u, z_t, u - t_s, t_t - t_s, dyn)
            else:
                L_bridge = torch.tensor(0.0, device=device)

            z0_cond = z_s.detach()
            cond_vec = torch.cat([z0_cond, time_emb(t_t)], dim=-1)
            logits = predictor(H_t, cond=cond_vec)
            L_mask = masked_cross_entropy(logits, x_t, mask_idx)

            L = args.l_mask * L_mask + args.l_inc * L_inc + args.l_bridge * L_bridge

            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            losses["mask"] += L_mask.item()
            losses["inc"] += L_inc.item()
            losses["bridge"] += L_bridge.item() if args.use_bridge else 0.0
            n_steps += 1

        print(
            f"[epoch {epoch}] L_mask={losses['mask'] / n_steps:.4f}  L_inc={losses['inc'] / n_steps:.4f}  L_bridge={losses['bridge'] / max(1, n_steps):.4f}"
        )

    print("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--toy", action="store_true")
    p.add_argument("--n_data", type=int, default=1024)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--vocab", type=int, default=128)
    p.add_argument("--mask_prob", type=float, default=0.35)

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_latent", type=int, default=32)
    p.add_argument("--t_dim", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dim_ff", type=int, default=1024)

    p.add_argument(
        "--process", type=str, default="brownian", choices=["brownian", "ou"]
    )
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--kappa", type=float, default=0.2)

    p.add_argument("--l_mask", type=float, default=1.0)
    p.add_argument("--l_inc", type=float, default=1.0)
    p.add_argument("--l_bridge", type=float, default=1.0)
    p.add_argument("--use_bridge", action="store_true")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
