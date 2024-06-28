import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random

from evo_diffusion.diffusion import DDIM, cosine_beta_schedule
from evo_diffusion.model_components import SinCosEmbedding
from evo_diffusion.training import (
    create_train_state,
    create_training_batches,
    train_step_ddim,
)


class DenoiseModel(nn.Module):
    sequence_length: int
    time_embedding_dim: int = 16

    def setup(self):
        self.time_embedding = SinCosEmbedding(self.time_embedding_dim)

    @nn.compact
    def __call__(self, x, t):
        t_embedded = self.time_embedding(t)

        x = x.reshape(x.shape[0], -1)

        for feat in [128, 128, 128]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)

        x = jnp.concatenate([x, t_embedded], axis=1)

        for feat in [128, 128, 128]:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        x = nn.Dense(self.sequence_length)(x)
        return x


# Generate synthetic data
def generate_gaussian_mixture(
    mus: list[float], sigmas: list[float], num_points_per_cluster: int
):
    noises = [
        np.random.normal(mu, sigma, size=(num_points_per_cluster, 1))
        for mu, sigma in zip(mus, sigmas)
    ]
    noises = np.concatenate((noises), axis=0)
    np.random.shuffle(noises)
    return noises


NUM_SAMPLES = 10_000  # Number of data points
BATCH_SIZE = 64  # Number of samples in each batch
NUM_DIFFUSION_STEPS = 20  # Number of steps for diffusion process
LEARNING_RATE = 4e-3  # Learning rate for optimizer
NUM_EPOCHS = 1000

if __name__ == "__main__":
    # Simulate data according to 2D Gaussian mixture
    np.random.seed(12)

    y = generate_gaussian_mixture([-10.0, 0.0, 10.0], [1.0, 0.5, 1.0], NUM_SAMPLES)
    y2 = generate_gaussian_mixture([-10.0, 0.0, 10.0], [1.0, 0.5, 1.0], NUM_SAMPLES)
    y = np.concatenate((y, y2), axis=1)

    # Get training data
    batches = create_training_batches(y, BATCH_SIZE)

    # Define denoising model and diffusion class
    beta_schedule = cosine_beta_schedule(NUM_DIFFUSION_STEPS, start=0.0001, stop=0.8)
    model = DenoiseModel(sequence_length=y.shape[-1])
    diffusion = DDIM(model, beta_schedule=beta_schedule)

    # Define optimizer and state
    rng = random.PRNGKey(0)
    rng, rng_init = random.split(rng)
    state = create_train_state(rng_init, model, LEARNING_RATE, y.shape[-1])

    # Train model
    train_step = jax.jit(train_step_ddim, static_argnums=0)
    losses = []
    for epoch in range(NUM_EPOCHS):
        rng, perm_rng, model_rng = jax.random.split(rng, 3)

        # Permute the batches
        perms = jax.random.permutation(perm_rng, len(batches))
        epoch_loss = []

        # For each batch, sample noising step and train model
        model_rngs = jax.random.split(model_rng, len(perms))
        for i, perm in enumerate(perms):
            t = np.random.choice(NUM_DIFFUSION_STEPS, size=(batches[perm].shape[0], 1))
            state, loss = train_step(diffusion, state, batches[perm], t, model_rngs[i])
            epoch_loss.append(loss)
        epoch_loss = np.array(epoch_loss).mean()
        losses.append(epoch_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {epoch_loss:.4f}")

    # Sample from model unconditionally
    samples = diffusion.reverse_process(
        rng, (3000, y.shape[-1]), {"params": state.params}
    )
