"""model architecture files."""

from .parallel_turbo_autoencoder import Turbo_AutoEncoder, generate_permutation_array

__all__ = [
    "Turbo_AutoEncoder",
    "generate_permutation_array",
]
