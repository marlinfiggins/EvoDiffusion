# Latent Evolution + Conditional Masked Prediction (Joint Training)
This repo provides a minimal PyTorch implementation of the model we discussed:

- **Encoder** `E_phi`: maps a (possibly masked) sequence to token embeddings `H` and a pooled vector `s`.
- **Projection** `P_xi`: maps the pooled vector to a **d-dimensional latent** `z` used for latent dynamics constraints.
- **Latent dynamics**: fixed Brownian or OU process to **shape the latent space** via increment (and optional bridge) likelihoods.
- **Masked predictor** `D_theta`: predicts **masked tokens** conditioned on **(z0, condition c)** using **FiLM** (AdaLayerNorm) inside each block.

**Joint training**: a single objective with
- masked prediction loss (generative head),
- latent increment (and optional bridge) likelihoods on the encoder-projected latents.

We **detach** `z0` only in the *conditioning path* (FiLM) so that the masked head can't distort the latent metrics. The dynamics losses still backprop through both the encoder and the projection head, to keep everything cooperative.

## Quick start

```bash
pip install -r requirements.txt
python train.py --toy --epochs 3
```

This runs on a small toy dataset (discrete tokens) to sanity-check shapes. Replace the dataset with your own sequences and time stamps.

## Training objective

L = λ_mask * L_mask + λ_inc * L_inc + λ_bridge * L_bridge

- Use `--use_bridge` if you have triplets `(0, u, T)`.
- Set `--process ou` to switch to OU dynamics.

## Sampling an ancestor

Given a descendant at time `t`:
1. Encode once: `z_t = P(E(x_t).s)`
2. Posterior `z0|zt` (Brownian or OU; see `model.py` for formulas).
3. Sample `z0` from that posterior.
4. Decode an ancestor by running the **masked predictor** on a fully masked template at condition `c=0` (iterative or single-shot).

## Notes
- The toy dataset uses synthetic sequences and simple time gaps; swap in real data loader with `(x_s, t_s), (x_t, t_t), dt, and masked x_t` (and triplets if available).
- The code is structured to be readable and hackable rather than hyper-optimized.
