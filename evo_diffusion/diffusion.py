import jax
from jax import random, lax
import jax.numpy as jnp
import numpy as np
class DDIM:
    def __init__(self, model, beta_schedule):
        """
        Initialize the DDIM class.

        Parameters:
        - model: A Flax model that takes an input tensor and a timestep and predicts the noise added at each step.
        - beta_schedule: A numpy array defining the variance schedule of the noise added during the forward process.
        """
        self.model = model  # Flax model to predict the noise
        self.beta_schedule = jnp.array(beta_schedule)
        self.alphas = 1.0 - self.beta_schedule
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1)

    def forward_process(self, key, x0, t):
        """
        Simulates the forward diffusion process for a given image and time step.

        Parameters:
        - key: A JAX random key.
        - x0: The original sample to noise.
        - t: The time step (index into the beta schedule) at which to sample the forward process.

        Returns:
        The noised image at time step t.
        """
        noise = random.normal(key, x0.shape)
        xt = self.sqrt_alphas_cumprod[t] * x0 + self.sqrt_one_minus_alphas_cumprod[t] * noise
        return xt, noise

    def single_step_denoise(self, noise, xt, t):
        """
        
        EQ: 9 in https://doi.org/10.48550/arXiv.2010.02502
        """
        return (xt - self.sqrt_one_minus_alphas_cumprod[t] * noise) / self.sqrt_alphas_cumprod[t]

    def reverse_process(self, key, shape, params, observed=None, score_fn=None, guidance_strength=1.0, num_steps=None):
        """
        Performs the guided reverse diffusion process using the model's noise predictions and optional observation guidance.

        Parameters:
        - key: A JAX random key for initial noise generation.
        - shape: The shape of the samples to generate.
        - observed: Optional observed data for guidance.
        - score_fn: Optional function that computes the guidance score given xt and observed data.
        - guidance_strength: A scalar adjusting the influence of the guidance score.
        - num_steps: Optional number of diffusion steps to use. Defaults to the length of the beta schedule.

        Returns:
        The reconstructed and potentially guided data after reversing the diffusion process.
        """
        if num_steps is None:
            num_steps = len(self.beta_schedule)
            
        if observed is not None:
            mask = np.isnan(observed)

        def _reverse_step(carry, i):
            xt, key = carry
            t = num_steps - i - 1

            # Predict noise at current step
            predicted_noise = self.model.apply(params, xt, jnp.array([t]))

            # Apply guidance if provided
            if observed is not None and score_fn is not None:
                def predict_obs_fn(xt, t):
                    return self.single_step_denoise(predicted_noise, xt, t)

                # Compute guidance from score function and update predicted noise
                guidance_score = score_fn(xt, jnp.array([t]), observed=observed, predict_obs_fn=predict_obs_fn)
                predicted_noise -= guidance_strength * guidance_score

            # Use the predicted noise to compute the denoised data (predicted observed)
            # EQ: 12 in https://doi.org/10.48550/arXiv.2010.02502
            alpha_prev = jnp.concatenate((jnp.ones(1), self.alphas_cumprod))[t]
            alpha = self.alphas_cumprod[t]
            
            pred_x0 = (xt - jnp.sqrt(1-alpha) * predicted_noise) / jnp.sqrt(alpha)
            direction_xt = jnp.sqrt(1-alpha_prev) * predicted_noise
            xt_minus_1 = jnp.sqrt(alpha_prev) * pred_x0 + direction_xt
            if observed is not None:
                xt_minus_1 = xt_minus_1.at[:, ~mask].set(observed[~mask])

            return (xt_minus_1, key), None

        # Sample noise and reverse step num_steps times
        xT = random.normal(key, shape)
        (xT_m_num_steps, _), _ = lax.scan(_reverse_step, (xT, key), jnp.arange(num_steps))
        return xT_m_num_steps
