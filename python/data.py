"""
Data loading, feature engineering, and dataset classes for Diagonal SSM.

Provides:
    - BybitClient: Fetches historical kline (candlestick) data from the Bybit v5 API.
    - StockDataLoader: Generates synthetic stock data for testing without external dependencies.
    - Feature engineering functions: returns, volatility, RSI, MACD, volume_change.
    - DiagonalSSMDataset: PyTorch Dataset that produces windowed feature sequences with targets.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kline data structure
# ---------------------------------------------------------------------------

@dataclass
class Kline:
    """Single candlestick (kline) bar.

    Attributes:
        timestamp: Unix timestamp in milliseconds.
        open: Opening price.
        high: Highest price.
        low: Lowest price.
        close: Closing price.
        volume: Trading volume.
    """
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# Bybit API client
# ---------------------------------------------------------------------------

class BybitClient:
    """Client for fetching kline data from the Bybit v5 REST API.

    Example::

        client = BybitClient()
        df = client.get_klines("BTCUSDT", interval="60", limit=500)
    """

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "60",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch historical kline data from Bybit.

        Args:
            symbol: Trading pair (e.g. 'BTCUSDT').
            interval: Kline interval in minutes. '1', '5', '15', '60', '240', 'D', 'W'.
            limit: Number of bars to retrieve (max 1000).
            start: Start timestamp in ms (optional).
            end: End timestamp in ms (optional).

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
            sorted by timestamp ascending.

        Raises:
            RuntimeError: If the API call fails.
        """
        import requests

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        logger.debug("Fetching Bybit klines: %s", params)
        resp = requests.get(self.BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error: {data.get('retMsg', 'unknown')}")

        rows = data["result"]["list"]
        klines = []
        for row in rows:
            klines.append(Kline(
                timestamp=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            ))

        # API returns newest first; reverse to chronological order
        klines.reverse()

        df = pd.DataFrame([vars(k) for k in klines])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        logger.debug("Fetched %d klines for %s", len(df), symbol)
        return df


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

class StockDataLoader:
    """Generates synthetic stock price data for offline testing.

    Uses geometric Brownian motion with configurable parameters to produce
    realistic-looking OHLCV data without requiring any external API or data source.
    """

    def generate(
        self,
        num_bars: int = 2000,
        initial_price: float = 100.0,
        drift: float = 0.0002,
        volatility: float = 0.02,
        seed: Optional[int] = 42,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data.

        Args:
            num_bars: Number of bars to generate.
            initial_price: Starting price.
            drift: Mean return per bar (mu * dt).
            volatility: Standard deviation of returns per bar (sigma * sqrt(dt)).
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with columns [open, high, low, close, volume] and a
            DatetimeIndex at hourly frequency.
        """
        rng = np.random.default_rng(seed)

        returns = rng.normal(drift, volatility, size=num_bars)
        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLCV from close prices
        opens = np.roll(prices, 1)
        opens[0] = initial_price
        intra_noise = rng.uniform(0.001, 0.01, size=num_bars)
        highs = np.maximum(opens, prices) * (1 + intra_noise)
        lows = np.minimum(opens, prices) * (1 - intra_noise)
        volumes = rng.lognormal(mean=10.0, sigma=1.0, size=num_bars)

        index = pd.date_range("2023-01-01", periods=num_bars, freq="h")

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        }, index=index)
        df.index.name = "timestamp"

        logger.debug("Generated %d synthetic bars", num_bars)
        return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """Log returns over a given period."""
    return np.log(close / close.shift(period))


def compute_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of log returns."""
    log_ret = compute_returns(close)
    return log_ret.rolling(window=window).std()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD histogram (MACD line minus signal line)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def compute_volume_change(volume: pd.Series, period: int = 1) -> pd.Series:
    """Percentage change in volume."""
    return volume.pct_change(periods=period)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features from an OHLCV DataFrame.

    Features computed:
        - returns: 1-bar log return of close
        - volatility: 20-bar rolling std of log returns
        - rsi: 14-bar RSI
        - macd: MACD histogram (12/26/9)
        - volume_change: 1-bar pct change of volume
        - close (normalized): close price normalized by rolling mean

    Args:
        df: DataFrame with columns [open, high, low, close, volume].

    Returns:
        DataFrame with computed feature columns, NaN rows dropped.
    """
    features = pd.DataFrame(index=df.index)
    features["returns"] = compute_returns(df["close"])
    features["volatility"] = compute_volatility(df["close"])
    features["rsi"] = compute_rsi(df["close"]) / 100.0  # normalize to [0, 1]
    features["macd"] = compute_macd(df["close"])
    features["volume_change"] = compute_volume_change(df["volume"])
    # Normalized close: deviation from 50-bar rolling mean
    rolling_mean = df["close"].rolling(window=50).mean()
    features["norm_close"] = (df["close"] - rolling_mean) / (rolling_mean + 1e-10)

    features.dropna(inplace=True)
    logger.debug("Engineered %d features over %d bars", features.shape[1], len(features))
    return features


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class DiagonalSSMDataset(Dataset):
    """Windowed time series dataset for Diagonal SSM training.

    Each sample is a tuple (features_window, target) where:
        - features_window: Tensor of shape (seq_len, num_features)
        - target: Tensor of shape (prediction_horizon,) containing future returns.

    Args:
        df: DataFrame with feature columns (output of engineer_features).
        close_prices: Series of close prices aligned with df index.
        seq_len: Number of past time steps in each input window.
        prediction_horizon: Number of future steps to predict.
        target_column: Feature column to use for generating targets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        close_prices: pd.Series,
        seq_len: int = 256,
        prediction_horizon: int = 24,
        target_column: str = "returns",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon

        # Align close prices to feature index
        close_aligned = close_prices.loc[df.index]

        # Standardize features (z-score)
        self.feature_mean = df.mean()
        self.feature_std = df.std().replace(0, 1)
        normalized = (df - self.feature_mean) / self.feature_std

        self.features = torch.tensor(normalized.values, dtype=torch.float32)
        self.close = torch.tensor(close_aligned.values, dtype=torch.float32)

        # Pre-compute target returns: cumulative return over prediction_horizon
        self.targets = self._compute_targets(close_aligned, prediction_horizon)

        # Valid indices
        self.valid_len = len(self.features) - seq_len - prediction_horizon
        if self.valid_len <= 0:
            raise ValueError(
                f"Not enough data: {len(self.features)} bars for seq_len={seq_len} + horizon={prediction_horizon}"
            )

        logger.debug(
            "DiagonalSSMDataset: %d samples, seq_len=%d, horizon=%d",
            self.valid_len, seq_len, prediction_horizon,
        )

    @staticmethod
    def _compute_targets(close: pd.Series, horizon: int) -> torch.Tensor:
        """Compute future log returns for each bar.

        For bar t, target[i] = log(close[t+i+1] / close[t+i]) for i in [0, horizon).
        """
        close_arr = close.values.astype(np.float64)
        n = len(close_arr)
        targets = np.zeros((n, horizon), dtype=np.float32)
        for i in range(horizon):
            future = np.roll(close_arr, -(i + 1))
            current = np.roll(close_arr, -i) if i > 0 else close_arr
            with np.errstate(divide="ignore", invalid="ignore"):
                ret = np.log(future / (current + 1e-10))
            # Zero out invalid tail entries
            ret[-(i + 1):] = 0.0
            if i > 0:
                ret[-i:] = 0.0
            targets[:, i] = ret.astype(np.float32)
        return torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return self.valid_len

    def __getitem__(self, idx):
        """Return (features_window, target_returns).

        Args:
            idx: Sample index.

        Returns:
            Tuple of (Tensor[seq_len, num_features], Tensor[prediction_horizon]).
        """
        x = self.features[idx: idx + self.seq_len]          # (seq_len, F)
        y = self.targets[idx + self.seq_len]                  # (horizon,)
        return x, y
