"""Training utilities."""

import torch


def save_checkpoint(model, optimizer, iteration, out):
    """Save checkpoint - to be implemented."""
    raise NotImplementedError("Checkpointing not yet implemented")


def load_checkpoint(src, model, optimizer):
    """Load checkpoint - to be implemented."""
    raise NotImplementedError("Checkpointing not yet implemented")