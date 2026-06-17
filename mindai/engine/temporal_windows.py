"""Cortical temporal integration — theta/gamma coupling.

Biological basis (Lisman & Jensen 2013; Buzsáki & Draguhn 2004):
  The brain integrates information across multiple timescales simultaneously
  using nested oscillations:

  Gamma (30–80 Hz) — encodes the current sensory moment.
    Each gamma cycle (~25 ms) corresponds to one discrete processing epoch.
    Modelled here as the primal impression (current input).

  Theta (4–8 Hz, ~125–250 ms) — groups sequences of gamma cycles.
    Theta phase determines which gamma cycles are potentiated.
    The retention buffer approximates theta-scale integration:
    past gamma epochs decay exponentially over ~5 ticks (≈ one theta cycle
    at 10 Hz tick rate).

  Prediction (top-down) — the brain constantly generates predictions about
    incoming input. When prediction matches input, surprise is low and
    processing is efficient. The protention term models this forward projection.

  The integrated 'thick present' (primal + retention + protention) is what
  enters consciousness — not a single moment but a ~300 ms window
  (Pöppel 1997 perceptual moment; Varela 1999 temporal binding).

Implementation:
  retention_buffer: ring buffer of past gamma epochs, exponentially weighted
  Decay τ = 2 ticks corresponds to ~200 ms at 10 Hz — within theta range.
  Weights: current=0.6, past retention=0.3, forward prediction=0.1.

Backward compatibility: class kept as HusserlianTime alias.
"""

import torch


class TemporalIntegration:
    """Theta/gamma-scale cortical temporal integration."""

    def __init__(self, num_nodes: int, window_size: int = 5, device: torch.device | None = None):
        self.device      = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes   = num_nodes
        self.window_size = window_size
        self.retention_buffer = torch.zeros(
            (window_size, num_nodes), device=self.device)
        # Exponential decay over theta window: τ = 2 ticks
        decay_weights = torch.exp(
            -torch.arange(window_size, device=self.device) / 2.0)
        self.decay_weights = decay_weights / decay_weights.sum()

    def create_conscious_now(
        self,
        primal_impression:  torch.Tensor,   # current gamma epoch
        protention_forecast: torch.Tensor,  # top-down prediction
    ) -> torch.Tensor:
        """Integrate current + past + predicted into one ~300 ms window."""
        self.retention_buffer = torch.roll(self.retention_buffer, shifts=1, dims=0)
        self.retention_buffer[0] = primal_impression
        retention_smear = torch.sum(
            self.retention_buffer * self.decay_weights.unsqueeze(1), dim=0)
        return (primal_impression * 0.6
                + retention_smear  * 0.3
                + protention_forecast * 0.1)


# Backward-compatible alias — brain.py imports HusserlianTime
HusserlianTime = TemporalIntegration
