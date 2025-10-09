import os
import pickle
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def simulate_genetic_sequences(
    num_sequences: int,
    sequence_length: int,
    time_span: int,
    fitness_fn: Callable,
    mutation_rate: float,
    time_dependent_fitness: bool = False,
    fix_initial_population: bool = False
):
    # Initialize the population with random sequences
    if fix_initial_population:
        initial_sequence = np.random.randint(0, 4, size=(sequence_length))
        population = np.vstack([initial_sequence] * num_sequences)
    else:
        population = np.random.randint(0, 4, size=(num_sequences, sequence_length))

    # Convert integers to nucleotides
    nucleotides = np.array(["A", "C", "G", "T"])

    sequences = nucleotides[population]

    # List to store sequences at all time steps
    all_sequences = [nucleotides[population]]

    for generation in range(time_span):
        # Calculate fitness for each sequence
        fitness = (
            fitness_fn(sequences, generation)
            if time_dependent_fitness
            else fitness_fn(sequences)
        )

        # Select parents based on fitness (using weighted random sampling)
        parent_indices = np.random.choice(
            num_sequences,
            size=num_sequences,
            p=np.exp(fitness) / np.sum(np.exp(fitness)),
        )
        parents = sequences[parent_indices]

        # Create offspring through mutation
        offspring = parents.copy()
        mutation_mask = np.random.random(offspring.shape) < mutation_rate
        offspring[mutation_mask] = nucleotides[
            np.random.randint(0, 4, size=int(np.sum(mutation_mask)))
        ]

        # Replace the population with the offspring
        sequences = offspring

        # Store the current generation's sequences
        all_sequences.append(sequences)

    return all_sequences


def example_fitness_function(population):
    # This is a simple fitness function that favors sequences with more 'A's
    return np.sum(population == "A", axis=1)


def int_to_char(sequences):
    seq_array = np.array([["ACGT".index(n) for n in seq] for seq in sequences])
    return seq_array


def char_to_int(sequences):
    # Create a lookup table
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    vectorized_lookup = np.vectorize(lookup.get)

    # Apply the lookup table to the entire array at once
    return vectorized_lookup(sequences)


def visualize_sequences(ax, sequences, num_sequences=10):
    # Define colors for each nucleotide
    color_map = {"A": "#ff9999", "C": "#99ff99", "G": "#9999ff", "T": "#ffff99"}
    cmap = ListedColormap([color_map[n] for n in "ACGT"])

    # Convert sequences to numeric representation
    num_sequences = min(num_sequences, len(sequences))
    seq_array = np.array(
        [["ACGT".index(n) for n in seq] for seq in sequences[:num_sequences]]
    )

    # Plot the sequences
    im = ax.imshow(seq_array, aspect="auto", cmap=cmap)

    # Add text for each nucleotide
    for i in range(seq_array.shape[0]):
        for j in range(seq_array.shape[1]):
            ax.text(j, i, sequences[i][j], ha="center", va="center", color="black")

    # Set labels and title
    ax.set_yticks(range(num_sequences))
    ax.set_yticklabels([f"Seq {i+1}" for i in range(num_sequences)])
    ax.set_xlabel("Nucleotide Position")
    ax.set_title("Genetic Sequences Visualization")

    # Add a color bar legend
    cbar = plt.colorbar(im, ax=ax, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar.set_ticklabels(["A", "C", "G", "T"])

    return ax


def visualize_sequence_metric(
    all_sequences,
    metric_function,
    ax=None,
    title="Sequence Metric Over Time",
    time_dependent_metric=False,
):
    """
    Visualize any scalar metric of the sequences over time.

    Parameters:
    - all_sequences: List of numpy arrays, each representing the population at a time step
    - metric_function: Function that takes a population array and returns a scalar
    - ax: Optinal axis to plot the metric on
    - title: Title for the plot

    Returns:
    - fig, ax: The figure and axis objects
    """
    time_steps = len(all_sequences)
    metric_values = []

    for generation, population in enumerate(all_sequences):
        metric_values.append(
            metric_function(population, generation)
            if time_dependent_metric
            else metric_function(population)
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    ax.plot(range(time_steps), metric_values)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.grid(True)

    return fig, ax


# Example metrics
def average_a_content(population):
    """Calculate the average A content across all sequences."""
    return np.mean(population == "A")


def gc_content(population):
    """Calculate the GC content across all sequences."""
    return np.mean((population == "G") | (population == "C"))


def mean_fitness(population):
    return example_fitness_function(population).mean()


# Saving and loading data
def pickle_sequences(all_sequences, filename, additional_data=None):
    """
    Pickle the simulated sequences and optionally additional data.

    Parameters:
    - all_sequences: List of numpy arrays, each representing the population at a time step
    - filename: String, the name of the file to save the pickled data
    - additional_data: Optional dictionary containing any additional data to save

    Returns:
    - None
    """
    # Ensure the filename ends with .pkl
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    # Prepare the data to be pickled
    data_to_pickle = {"sequences": all_sequences, "additional_data": additional_data}

    # Pickle the data
    with open(filename, "wb") as f:
        pickle.dump(data_to_pickle, f)

    print(f"Sequences pickled successfully to {filename}")


def load_pickled_sequences(filename):
    """
    Load pickled sequences and additional data.

    Parameters:
    - filename: String, the name of the file to load the pickled data from

    Returns:
    - Tuple (all_sequences, additional_data)
    """
    # Ensure the filename ends with .pkl
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Load the pickled data
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)

    all_sequences = loaded_data["sequences"]
    additional_data = loaded_data["additional_data"]

    print(f"Sequences loaded successfully from {filename}")

    return all_sequences, additional_data
