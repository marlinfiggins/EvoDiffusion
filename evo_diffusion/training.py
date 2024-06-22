import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


def create_train_state(key, model, learning_rate, input_dim):
    init_dim = (1, input_dim) if isinstance(input_dim, int) else (1,) + input_dim
    params = model.init(key, jnp.ones(init_dim), jnp.ones([1]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_step_ddim(diffusion, state, batch, t, key):
    # Noise the batch
    xt, noise = diffusion.forward_process(key, batch, t)

    def loss_fn(params):
        # Sample predicted noise from noised input
        pred_noise = state.apply_fn({"params": params}, xt, t)
        # Compute MSE between predicted and true noise
        return jnp.mean(jnp.square(pred_noise - noise))

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

