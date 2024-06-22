import flax.linen as nn
import jax.numpy as jnp


class SinCosEmbedding(nn.Module):
    embedding_dim: int  # Dimensionality of the sinusoidal embedding

    def setup(self):
        # Create frequencies for the SinCos embedding
        self.frequencies = jnp.array(
            [
                10000 ** (-2 * (i // 2) / self.embedding_dim)
                for i in range(self.embedding_dim)
            ]
        )

    def __call__(self, t):
        # Expand t if it's not already expanded
        t = jnp.expand_dims(t, -1)
        # Apply sinusoidal embedding
        embeddings = jnp.concatenate(
            [jnp.sin(t * self.frequencies), jnp.cos(t * self.frequencies)], axis=-1
        )
        # Reshape to ensure the output is (batch_size, embedding_dim)
        return embeddings.reshape(t.shape[0], -1)
