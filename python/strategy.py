"""
Trading strategy, signal generation, and backtesting for Diagonal SSM predictions.

Provides:
    - SignalType: Enum for BUY/SELL/HOLD signals.
    - Signal: Dataclass associating a signal with a timestamp and confidence.
    - BacktestConfig: Dataclass with backtesting parameters (capital, commissions, etc.).
    - BacktestResult: Dataclass with performance metrics and equity curve.
    - Backtester: Generates trading signals from model predictions, executes a
      vectorized backtest with position management, and computes risk metrics.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal types and data structures
# ---------------------------------------------------------------------------

class SignalType(Enum):
    """Trading signal direction."""
    BUY = 1
    HOLD = 0
    SELL = -1


@dataclass
class Signal:
    """A trading signal produced by the model.

    Attributes:
        timestamp: Time at which the signal was generated.
        signal_type: Direction of the trade (BUY, SELL, HOLD).
        confidence: Strength of the signal in [0, 1].
        predicted_return: Raw model prediction (cumulative expected return).
    """
    timestamp: pd.Timestamp
    signal_type: SignalType
    confidence: float
    predicted_return: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine.

    Attributes:
        initial_capital: Starting portfolio value.
        commission: Commission per trade as a fraction of trade value.
        slippage: Slippage per trade as a fraction of price.
        buy_threshold: Minimum predicted return to trigger a BUY.
        sell_threshold: Maximum predicted return (negative) to trigger a SELL.
        stop_loss: Maximum allowed drawdown per position before forced exit.
        max_position_size: Maximum fraction of capital in a single position.
        annualization_factor: Number of bars per year for Sharpe/Sortino (e.g. 8760 for hourly).
    """
    initial_capital: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    buy_threshold: float = 0.001
    sell_threshold: float = -0.001
    stop_loss: float = 0.05
    max_position_size: float = 1.0
    annualization_factor: float = 8760.0  # hourly bars


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Results of a backtest run.

    Attributes:
        total_return: Cumulative return over the backtest period.
        sharpe_ratio: Annualized Sharpe ratio.
        sortino_ratio: Annualized Sortino ratio (downside deviation only).
        max_drawdown: Maximum peak-to-trough drawdown.
        win_rate: Fraction of winning trades.
        num_trades: Total number of completed round-trip trades.
        equity_curve: Series of portfolio values over time.
        signals: List of all generated signals.
    """
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    equity_curve: pd.Series
    signals: List[Signal] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of the backtest."""
        return (
            f"Backtest Results\n"
            f"  Total Return:   {self.total_return:+.4f} ({self.total_return * 100:+.2f}%)\n"
            f"  Sharpe Ratio:   {self.sharpe_ratio:.4f}\n"
            f"  Sortino Ratio:  {self.sortino_ratio:.4f}\n"
            f"  Max Drawdown:   {self.max_drawdown:.4f} ({self.max_drawdown * 100:.2f}%)\n"
            f"  Win Rate:       {self.win_rate:.4f} ({self.win_rate * 100:.1f}%)\n"
            f"  Num Trades:     {self.num_trades}\n"
        )


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """Backtesting engine for Diagonal SSM trading strategies.

    Workflow:
        1. Generate signals from model predictions via `generate_signals`.
        2. Run backtest via `run` using those signals and close prices.
        3. Inspect the returned BacktestResult.

    Args:
        config: BacktestConfig with parameters.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def generate_signals(
        self,
        predictions: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> List[Signal]:
        """Convert model predictions into trading signals.

        The predicted return is taken as the mean of the prediction horizon vector.
        A BUY is generated when predicted return > buy_threshold,
        a SELL when predicted return < sell_threshold, else HOLD.

        Args:
            predictions: Array of shape (N, prediction_horizon) with predicted future returns.
            timestamps: DatetimeIndex aligned with predictions.

        Returns:
            List of Signal objects.
        """
        cfg = self.config
        signals = []

        for i in range(len(predictions)):
            mean_pred = float(np.mean(predictions[i]))
            abs_pred = abs(mean_pred)

            if mean_pred > cfg.buy_threshold:
                sig_type = SignalType.BUY
            elif mean_pred < cfg.sell_threshold:
                sig_type = SignalType.SELL
            else:
                sig_type = SignalType.HOLD

            # Confidence: how far prediction is from threshold (clipped to [0, 1])
            if sig_type == SignalType.BUY:
                confidence = min(abs_pred / (cfg.buy_threshold * 10 + 1e-10), 1.0)
            elif sig_type == SignalType.SELL:
                confidence = min(abs_pred / (abs(cfg.sell_threshold) * 10 + 1e-10), 1.0)
            else:
                confidence = 0.0

            signals.append(Signal(
                timestamp=timestamps[i],
                signal_type=sig_type,
                confidence=confidence,
                predicted_return=mean_pred,
            ))

        n_buy = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        n_sell = sum(1 for s in signals if s.signal_type == SignalType.SELL)
        n_hold = sum(1 for s in signals if s.signal_type == SignalType.HOLD)
        logger.debug("Signals generated: %d BUY, %d SELL, %d HOLD", n_buy, n_sell, n_hold)

        return signals

    def run(
        self,
        signals: List[Signal],
        close_prices: pd.Series,
    ) -> BacktestResult:
        """Execute the backtest with position management.

        Simulates a single-asset long/short strategy:
            - On BUY: enter long (if flat or short, close existing and go long).
            - On SELL: enter short (if flat or long, close existing and go short).
            - On HOLD: maintain current position.
            - Stop-loss: close position if unrealized loss exceeds stop_loss.

        Args:
            signals: List of Signal objects (must be chronologically ordered).
            close_prices: Series of close prices indexed by timestamp.

        Returns:
            BacktestResult with all performance metrics.
        """
        cfg = self.config
        capital = cfg.initial_capital
        position = 0.0        # number of units held (positive=long, negative=short)
        entry_price = 0.0
        equity_values = []
        equity_timestamps = []
        trades = []            # list of (entry_price, exit_price, direction)

        for signal in signals:
            ts = signal.timestamp
            if ts not in close_prices.index:
                continue

            price = float(close_prices.loc[ts])
            if price <= 0:
                continue

            # Check stop-loss on existing position
            if position != 0.0:
                if position > 0:
                    unrealized_pnl = (price - entry_price) / entry_price
                else:
                    unrealized_pnl = (entry_price - price) / entry_price

                if unrealized_pnl < -cfg.stop_loss:
                    # Close position (stop-loss)
                    exit_cost = abs(position) * price * (cfg.commission + cfg.slippage)
                    if position > 0:
                        capital += position * price - exit_cost
                        trades.append((entry_price, price, 1))
                    else:
                        capital += abs(position) * (2 * entry_price - price) - exit_cost
                        trades.append((entry_price, price, -1))
                    position = 0.0

            # Process signal
            if signal.signal_type == SignalType.BUY and position <= 0:
                # Close short if any
                if position < 0:
                    exit_cost = abs(position) * price * (cfg.commission + cfg.slippage)
                    capital += abs(position) * (2 * entry_price - price) - exit_cost
                    trades.append((entry_price, price, -1))
                    position = 0.0

                # Open long
                invest = capital * cfg.max_position_size
                entry_cost = invest * (cfg.commission + cfg.slippage)
                units = (invest - entry_cost) / price
                position = units
                entry_price = price
                capital -= invest

            elif signal.signal_type == SignalType.SELL and position >= 0:
                # Close long if any
                if position > 0:
                    exit_cost = position * price * (cfg.commission + cfg.slippage)
                    capital += position * price - exit_cost
                    trades.append((entry_price, price, 1))
                    position = 0.0

                # Open short
                invest = capital * cfg.max_position_size
                entry_cost = invest * (cfg.commission + cfg.slippage)
                units = (invest - entry_cost) / price
                position = -units
                entry_price = price
                capital -= invest

            # Mark-to-market equity
            if position > 0:
                equity = capital + position * price
            elif position < 0:
                equity = capital + abs(position) * (2 * entry_price - price)
            else:
                equity = capital

            equity_values.append(equity)
            equity_timestamps.append(ts)

        # Close any remaining position at last price
        if position != 0.0 and len(signals) > 0:
            last_price = float(close_prices.iloc[-1])
            if position > 0:
                exit_cost = position * last_price * (cfg.commission + cfg.slippage)
                capital += position * last_price - exit_cost
                trades.append((entry_price, last_price, 1))
            else:
                exit_cost = abs(position) * last_price * (cfg.commission + cfg.slippage)
                capital += abs(position) * (2 * entry_price - last_price) - exit_cost
                trades.append((entry_price, last_price, -1))

        # Build equity curve
        equity_curve = pd.Series(equity_values, index=equity_timestamps, name="equity")

        # Compute metrics
        total_return = (equity_curve.iloc[-1] / cfg.initial_capital - 1.0) if len(equity_curve) > 0 else 0.0

        # Returns for Sharpe/Sortino
        if len(equity_curve) > 1:
            equity_returns = equity_curve.pct_change().dropna()
            mean_ret = equity_returns.mean()
            std_ret = equity_returns.std()
            downside_ret = equity_returns[equity_returns < 0]
            downside_std = downside_ret.std() if len(downside_ret) > 0 else 1e-10

            ann = np.sqrt(cfg.annualization_factor)
            sharpe = (mean_ret / (std_ret + 1e-10)) * ann
            sortino = (mean_ret / (downside_std + 1e-10)) * ann
        else:
            sharpe = 0.0
            sortino = 0.0

        # Max drawdown
        if len(equity_curve) > 0:
            running_max = equity_curve.cummax()
            drawdowns = (equity_curve - running_max) / running_max
            max_drawdown = abs(float(drawdowns.min()))
        else:
            max_drawdown = 0.0

        # Win rate
        wins = 0
        for entry_p, exit_p, direction in trades:
            pnl = (exit_p - entry_p) * direction
            if pnl > 0:
                wins += 1
        win_rate = wins / len(trades) if trades else 0.0

        result = BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_trades=len(trades),
            equity_curve=equity_curve,
            signals=signals,
        )

        logger.debug("Backtest complete: %d trades, return=%.4f", len(trades), total_return)
        return result
