"""
Diagonal State Space Model (Diagonal SSM) for financial time series forecasting.

Implements the diagonal parameterization of state space models following:
"Diagonal State Spaces are as Effective as Structured State Spaces"
(Gupta, Gu, Berant, 2022 - https://arxiv.org/abs/2203.14343)

The continuous-time SSM is defined as:
    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)

where A is constrained to be diagonal with complex eigenvalues.

Key insight: By restricting A = diag(lambda_1, ..., lambda_N), the SSM
can be computed efficiently via element-wise operations rather than
matrix multiplications, while retaining the expressivity of S4.

Discretization uses the Zero-Order Hold (ZOH) method:
    A_bar = exp(Lambda * dt)
    B_bar = (A_bar - I) * Lambda^{-1} * B

The convolution kernel is computed via the Vandermonde product:
    K[l] = Re(sum_i C_i * A_bar_i^l * B_bar_i)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DiagonalSSMConfig:
    """Configuration for the Diagonal SSM model.

    Attributes:
        input_features: Number of input features per time step (e.g. OHLCV + indicators).
        state_dim: Dimension of the latent state (N). Each layer has N complex eigenvalues.
        d_model: Hidden dimension of the model (H).
        num_layers: Number of stacked DiagonalSSM layers.
        dropout: Dropout probability applied after each layer.
        init_method: Initialization for eigenvalues. One of 's4d_lin', 'hippo', 'random'.
        bidirectional: If True, use forward + backward SSM and merge outputs.
        prediction_horizon: Number of future time steps to predict.
        lr_dt: If True, use a learnable log-scale step size (dt) per layer.
        seq_len: Maximum sequence length for kernel computation.
        dt_min: Minimum value for the discretization step size.
        dt_max: Maximum value for the discretization step size.
    """
    input_features: int = 6
    state_dim: int = 64
    d_model: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    init_method: str = "s4d_lin"
    bidirectional: bool = False
    prediction_horizon: int = 24
    lr_dt: bool = True
    seq_len: int = 256
    dt_min: float = 0.001
    dt_max: float = 0.1


class DiagonalSSMKernel(nn.Module):
    """Diagonal SSM kernel with complex-valued diagonal state matrix.

    The kernel transforms an input sequence u of length L into an output
    sequence y of the same length via a causal convolution with a learned
    kernel K of length L.

    Parameters are stored in complex64 for memory efficiency.

    Args:
        d_model: Hidden dimension H (number of independent SSM copies).
        state_dim: State dimension N (number of eigenvalues per copy).
        seq_len: Maximum sequence length L.
        init_method: Eigenvalue initialization ('s4d_lin', 'hippo', 'random').
        lr_dt: Whether to learn the discretization step size.
        dt_min: Minimum discretization step size.
        dt_max: Maximum discretization step size.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        seq_len: int = 256,
        init_method: str = "s4d_lin",
        lr_dt: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.seq_len = seq_len

        # Initialize parameters
        log_neg_real, imag = self._initialize_params(d_model, state_dim, init_method)
        self.log_neg_real = nn.Parameter(log_neg_real)  # (H, N)
        self.imag = nn.Parameter(imag)                  # (H, N)

        # B and C are complex-valued: (H, N)
        self.B_re = nn.Parameter(torch.randn(d_model, state_dim) * 0.5)
        self.B_im = nn.Parameter(torch.randn(d_model, state_dim) * 0.5)
        self.C_re = nn.Parameter(torch.randn(d_model, state_dim) * 0.5)
        self.C_im = nn.Parameter(torch.randn(d_model, state_dim) * 0.5)

        # Learnable log step size
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        if lr_dt:
            self.log_dt = nn.Parameter(log_dt)
        else:
            self.register_buffer("log_dt", log_dt)

        # Feed-through parameter D
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)

        logger.debug(
            "DiagonalSSMKernel initialized: d_model=%d, state_dim=%d, init=%s",
            d_model, state_dim, init_method,
        )

    @staticmethod
    def _initialize_params(d_model: int, state_dim: int, method: str):
        """Initialize the diagonal eigenvalue parameters.

        The eigenvalues are parameterized as:
            lambda_n = -exp(log_neg_real_n) + j * imag_n

        This ensures Re(lambda) < 0 for stability.

        Args:
            d_model: Number of independent SSM copies.
            state_dim: Number of eigenvalues per copy.
            method: Initialization strategy.

        Returns:
            Tuple of (log_neg_real, imag) tensors each of shape (d_model, state_dim).
        """
        if method == "s4d_lin":
            # S4D-Lin: lambda_n = -1/2 + j*pi*n  (n = 0, ..., N-1)
            # log_neg_real = log(1/2) for all
            log_neg_real = torch.full((d_model, state_dim), math.log(0.5))
            imag = math.pi * torch.arange(state_dim).float().unsqueeze(0).expand(d_model, -1)
            logger.debug("S4D-Lin init: Re(lambda)=-0.5, Im(lambda)=pi*n")
        elif method == "hippo":
            # HiPPO-inspired: lambda_n = -(n+1/2) + j*pi*(n+1/2)
            n = torch.arange(state_dim).float()
            real_part = n + 0.5
            log_neg_real = torch.log(real_part).unsqueeze(0).expand(d_model, -1)
            imag = (math.pi * real_part).unsqueeze(0).expand(d_model, -1)
            logger.debug("HiPPO init: Re(lambda)=-(n+0.5), Im(lambda)=pi*(n+0.5)")
        elif method == "random":
            log_neg_real = torch.randn(d_model, state_dim) * 0.5
            imag = torch.randn(d_model, state_dim) * math.pi
            logger.debug("Random init for eigenvalues")
        else:
            raise ValueError(f"Unknown init method: {method}. Use 's4d_lin', 'hippo', or 'random'.")

        return log_neg_real, imag

    def _get_lambda(self):
        """Compute complex eigenvalues from parameterization.

        Returns:
            Complex tensor of shape (H, N) with Re < 0 guaranteed.
        """
        neg_real = -torch.exp(self.log_neg_real)  # guaranteed negative
        return torch.complex(neg_real, self.imag)  # (H, N)

    def discretize(self, Lambda, dt):
        """Zero-Order Hold (ZOH) discretization.

        Given continuous parameters Lambda, B:
            A_bar = exp(Lambda * dt)
            B_bar = (A_bar - I) * Lambda^{-1} * B

        Args:
            Lambda: Complex eigenvalues (H, N).
            dt: Step sizes (H, 1).

        Returns:
            Tuple (A_bar, B_bar) each of shape (H, N).
        """
        B = torch.complex(self.B_re, self.B_im)  # (H, N)

        # A_bar = exp(Lambda * dt)
        A_bar = torch.exp(Lambda * dt)  # (H, N)

        # B_bar = (A_bar - I) * Lambda^{-1} * B
        # Numerically: (exp(Lambda*dt) - 1) / Lambda * B
        Lambda_inv = 1.0 / Lambda
        B_bar = (A_bar - 1.0) * Lambda_inv * B  # (H, N)

        return A_bar, B_bar

    def compute_kernel(self, L: Optional[int] = None):
        """Compute the SSM convolution kernel of length L.

        K[l] = Re( sum_n C_n * A_bar_n^l * B_bar_n ),  l = 0, ..., L-1

        This is computed via the Vandermonde product for efficiency.

        Args:
            L: Kernel length. Defaults to self.seq_len.

        Returns:
            Real tensor of shape (H, L).
        """
        if L is None:
            L = self.seq_len

        Lambda = self._get_lambda()  # (H, N)
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (H, 1)

        A_bar, B_bar = self.discretize(Lambda, dt)
        C = torch.complex(self.C_re, self.C_im)  # (H, N)

        # Vandermonde product: K[l] = C * A_bar^l * B_bar summed over N
        # Powers: A_bar^l for l = 0, ..., L-1
        # Shape: (H, N, L)
        powers = torch.arange(L, device=A_bar.device, dtype=torch.float32)
        # A_bar^l = exp(l * log(A_bar))
        log_A_bar = torch.log(A_bar)  # (H, N)
        A_powers = torch.exp(log_A_bar.unsqueeze(-1) * powers.unsqueeze(0).unsqueeze(0))  # (H, N, L)

        # K = Re( sum_n C_n * B_bar_n * A_bar_n^l )
        CB = (C * B_bar).unsqueeze(-1)  # (H, N, 1)
        K = torch.sum(CB * A_powers, dim=1).real  # (H, L)

        return K

    def forward(self, u):
        """Apply the SSM kernel to input sequence via FFT-based convolution.

        Args:
            u: Input tensor of shape (B, H, L).

        Returns:
            Output tensor of shape (B, H, L).
        """
        B_size, H, L = u.shape

        # Compute kernel
        K = self.compute_kernel(L)  # (H, L)

        # FFT-based causal convolution
        # Pad to avoid circular convolution artifacts
        fft_len = 2 * L
        u_f = torch.fft.rfft(u.float(), n=fft_len, dim=-1)       # (B, H, fft_len//2+1)
        K_f = torch.fft.rfft(K.float(), n=fft_len, dim=-1)       # (H, fft_len//2+1)
        y_f = u_f * K_f.unsqueeze(0)                               # (B, H, fft_len//2+1)
        y = torch.fft.irfft(y_f, n=fft_len, dim=-1)[..., :L]      # (B, H, L)

        # Add feed-through: y += D * u
        y = y + self.D.unsqueeze(0).unsqueeze(-1) * u.float()

        return y


class DiagonalSSMLayer(nn.Module):
    """Single Diagonal SSM layer with projections, gating, and normalization.

    Architecture:
        input -> LayerNorm -> input_proj -> SSM kernel -> GLU gate -> output_proj -> dropout -> + residual

    Args:
        config: DiagonalSSMConfig with model hyperparameters.
    """

    def __init__(self, config: DiagonalSSMConfig):
        super().__init__()
        d = config.d_model

        self.norm = nn.LayerNorm(d)
        self.input_proj = nn.Linear(d, d)

        self.ssm = DiagonalSSMKernel(
            d_model=d,
            state_dim=config.state_dim,
            seq_len=config.seq_len,
            init_method=config.init_method,
            lr_dt=config.lr_dt,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
        )

        # GLU gating: project to 2*d, split, apply sigmoid gate
        self.gate_proj = nn.Linear(d, 2 * d)
        self.output_proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(config.dropout)

        self.bidirectional = config.bidirectional
        if self.bidirectional:
            self.ssm_back = DiagonalSSMKernel(
                d_model=d,
                state_dim=config.state_dim,
                seq_len=config.seq_len,
                init_method=config.init_method,
                lr_dt=config.lr_dt,
                dt_min=config.dt_min,
                dt_max=config.dt_max,
            )
            self.merge_proj = nn.Linear(2 * d, d)

    def forward(self, x):
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (B, L, H).

        Returns:
            Output tensor of shape (B, L, H).
        """
        residual = x
        x = self.norm(x)

        # Input projection
        x = self.input_proj(x)  # (B, L, H)

        # Transpose for SSM: (B, L, H) -> (B, H, L)
        u = x.transpose(1, 2)

        # Forward SSM
        y_fwd = self.ssm(u)  # (B, H, L)

        if self.bidirectional:
            # Backward SSM: flip, apply, flip back
            u_back = torch.flip(u, dims=[-1])
            y_back = self.ssm_back(u_back)
            y_back = torch.flip(y_back, dims=[-1])
            y = torch.cat([y_fwd, y_back], dim=1)  # (B, 2H, L)
            y = y.transpose(1, 2)  # (B, L, 2H)
            y = self.merge_proj(y)  # (B, L, H)
        else:
            y = y_fwd.transpose(1, 2)  # (B, L, H)

        # GLU activation
        gate_input = self.gate_proj(y)  # (B, L, 2H)
        a, b = gate_input.chunk(2, dim=-1)
        y = a * torch.sigmoid(b)  # (B, L, H)

        # Output projection + dropout + residual
        y = self.output_proj(y)
        y = self.dropout(y)
        y = y + residual

        return y


class DiagonalSSMForecaster(nn.Module):
    """Diagonal SSM model for financial time series forecasting.

    Stacks multiple DiagonalSSMLayer blocks with a final regression head
    that maps the last hidden state to a prediction_horizon-length output.

    Args:
        config: DiagonalSSMConfig with all hyperparameters.
    """

    def __init__(self, config: DiagonalSSMConfig):
        super().__init__()
        self.config = config

        # Input embedding: project raw features to d_model
        self.input_embed = nn.Linear(config.input_features, config.d_model)

        # Stack of Diagonal SSM layers
        self.layers = nn.ModuleList([
            DiagonalSSMLayer(config) for _ in range(config.num_layers)
        ])

        # Prediction head: use last time-step hidden state for forecasting
        self.head_norm = nn.LayerNorm(config.d_model)
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.prediction_horizon),
        )

        self._init_weights()
        logger.debug(
            "DiagonalSSMForecaster: %d layers, d_model=%d, state_dim=%d, horizon=%d",
            config.num_layers, config.d_model, config.state_dim, config.prediction_horizon,
        )

    def _init_weights(self):
        """Initialize linear layer weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, L, input_features).

        Returns:
            Predictions of shape (B, prediction_horizon).
        """
        # Embed input features
        h = self.input_embed(x)  # (B, L, d_model)

        # Pass through SSM layers
        for layer in self.layers:
            h = layer(h)

        # Use last time step for prediction
        h_last = h[:, -1, :]  # (B, d_model)
        h_last = self.head_norm(h_last)

        return self.prediction_head(h_last)  # (B, prediction_horizon)

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
