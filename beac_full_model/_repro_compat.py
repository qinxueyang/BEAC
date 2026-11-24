"""Small compatibility helpers that preserve historical RNG state."""

from torch import nn


def advance_legacy_disabled_init_rng(feature_dim: int) -> None:
    """Match initialization RNG draws from modules absent in the BEAC full-model graph.

    The historical implementation instantiated two unused Support-Gate
    projections (``Linear(D, D)``), then a disabled semantic enhancer
    (``Linear(2D, D)`` and ``Linear(D, D)``). Their initialization advanced
    the global torch RNG before DataLoader workers were created. Preserve
    those draws without retaining dead parameters in the BEAC full model model.
    """

    nn.Linear(feature_dim, feature_dim)
    nn.Linear(feature_dim, feature_dim)
    nn.Linear(feature_dim * 2, feature_dim)
    nn.Linear(feature_dim, feature_dim)
