"""
Chapter 138: Diagonal State Space Models (Diagonal SSM) for Financial Time Series.

This module implements diagonal SSMs based on the paper:
"Diagonal State Spaces are as Effective as Structured State Spaces"
(Gupta, Gu, Berant, 2022 - https://arxiv.org/abs/2203.14343)

Diagonal SSMs simplify the S4 (Structured State Space) framework by replacing
dense state matrices with diagonal ones, achieving comparable performance with
significantly reduced computational complexity. This makes them well-suited
for high-frequency financial time series prediction.

Key components:
    - model: DiagonalSSMForecaster and supporting kernel/layer classes
    - data: Data loading (Bybit API + synthetic), feature engineering, dataset
    - strategy: Backtesting framework with signal generation and performance metrics
    - example_usage: End-to-end demonstration script
"""

from .model import DiagonalSSMConfig, DiagonalSSMKernel, DiagonalSSMLayer, DiagonalSSMForecaster
from .data import BybitClient, StockDataLoader, DiagonalSSMDataset
from .strategy import Backtester, BacktestConfig, BacktestResult, Signal, SignalType

__all__ = [
    "DiagonalSSMConfig",
    "DiagonalSSMKernel",
    "DiagonalSSMLayer",
    "DiagonalSSMForecaster",
    "BybitClient",
    "StockDataLoader",
    "DiagonalSSMDataset",
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "Signal",
    "SignalType",
]
