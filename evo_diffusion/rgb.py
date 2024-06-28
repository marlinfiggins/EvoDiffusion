import matplotlib.pyplot as plt
import numpy as np


def sample_rgb_sequences(num_samples, sequence_length):
    """
    Generate a batch of RGB sequences.

    Parameters:
    - num_samples (N): Number of sequences to sample.
    - sequence_length (L): Length of each sequence.

    Returns:
    - numpy array of shape (N, L, 3) with RGB values.
    """
    x = np.random.uniform(size=(num_samples, sequence_length, 3))
    x = np.random.randint(0, 3, size=(num_samples, sequence_length))
    x = (x[:, :, None] == np.arange(3)).astype(float)
    return x


def visualize_rgb_sequences(rgb_sequences):
    """
    Visualize RGB sequences using matplotlib.

    Parameters:
    - rgb_sequences: numpy array of shape (B, L, 3) with RGB values.
    """
    batch_size, sequence_length, _ = rgb_sequences.shape

    _, axes = plt.subplots(batch_size, 1, figsize=(sequence_length, batch_size))

    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        ax = axes[i]
        ax.imshow([rgb_sequences[i]], aspect="auto")
        ax.axis("off")

    plt.show()
