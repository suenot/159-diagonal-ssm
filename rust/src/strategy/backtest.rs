//! Backtesting engine for evaluating trading strategies.

use anyhow::Result;
use log::{debug, info};
use serde::{Deserialize, Serialize};

use super::signals::Signal;

/// Configuration for the backtesting engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital.
    pub initial_capital: f64,
    /// Transaction cost as a fraction of trade value.
    pub transaction_cost: f64,
    /// Position sizing fraction (1.0 = all-in).
    pub position_size: f64,
    /// Risk-free rate for Sharpe ratio calculation (annualized).
    pub risk_free_rate: f64,
    /// Number of trading periods per year (for annualization).
    pub periods_per_year: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,
            position_size: 1.0,
            risk_free_rate: 0.02,
            periods_per_year: 252.0,
        }
    }
}

/// Results from a backtesting run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Total return as a fraction (e.g. 0.15 = 15%).
    pub total_return: f64,
    /// Annualized return.
    pub annualized_return: f64,
    /// Sharpe ratio (annualized).
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized, using downside deviation).
    pub sortino_ratio: f64,
    /// Maximum drawdown as a fraction.
    pub max_drawdown: f64,
    /// Win rate (fraction of profitable trades).
    pub win_rate: f64,
    /// Total number of trades executed.
    pub total_trades: usize,
    /// Profit factor (gross profits / gross losses).
    pub profit_factor: f64,
    /// Calmar ratio (annualized return / max drawdown).
    pub calmar_ratio: f64,
    /// Final portfolio value.
    pub final_value: f64,
    /// Equity curve (portfolio value at each step).
    pub equity_curve: Vec<f64>,
    /// Per-period returns.
    pub returns: Vec<f64>,
}

/// Backtesting engine that simulates trading based on signals and price data.
#[derive(Debug, Clone)]
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester with the given configuration.
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Create a backtester with default configuration.
    pub fn default_config() -> Self {
        Self {
            config: BacktestConfig::default(),
        }
    }

    /// Run the backtest with the given signals and price returns.
    ///
    /// # Arguments
    ///
    /// * `signals` - Trading signals for each period
    /// * `returns` - Actual price returns for each period (log returns)
    ///
    /// # Returns
    ///
    /// A [`BacktestResult`] with all performance metrics.
    pub fn run(&self, signals: &[Signal], returns: &[f64]) -> Result<BacktestResult> {
        let n = signals.len().min(returns.len());
        if n == 0 {
            anyhow::bail!("No data to backtest");
        }

        info!("Running backtest on {} periods", n);

        let mut equity = self.config.initial_capital;
        let mut equity_curve = Vec::with_capacity(n + 1);
        let mut portfolio_returns = Vec::with_capacity(n);
        let mut peak_equity = equity;
        let mut max_drawdown = 0.0_f64;
        let mut total_trades = 0;
        let mut winning_trades = 0;
        let mut gross_profit = 0.0;
        let mut gross_loss = 0.0;
        let mut prev_signal = Signal::Flat;

        equity_curve.push(equity);

        for i in 0..n {
            let signal = signals[i];
            let ret = returns[i];

            // Count trade if signal changes
            if signal != prev_signal {
                total_trades += 1;
                // Apply transaction cost
                equity *= 1.0 - self.config.transaction_cost;
            }

            // Compute portfolio return based on signal
            let position = match signal {
                Signal::Long => self.config.position_size,
                Signal::Short => -self.config.position_size,
                Signal::Flat => 0.0,
            };

            let period_return = position * ret;
            let pnl = equity * period_return;

            if pnl > 0.0 {
                gross_profit += pnl;
                if signal != prev_signal && signal != Signal::Flat {
                    winning_trades += 1;
                }
            } else if pnl < 0.0 {
                gross_loss += pnl.abs();
            }

            equity *= 1.0 + period_return;
            equity_curve.push(equity);
            portfolio_returns.push(period_return);

            // Update drawdown
            if equity > peak_equity {
                peak_equity = equity;
            }
            let drawdown = (peak_equity - equity) / peak_equity;
            max_drawdown = max_drawdown.max(drawdown);

            prev_signal = signal;
        }

        // Calculate metrics
        let total_return = (equity - self.config.initial_capital) / self.config.initial_capital;
        let n_periods = n as f64;
        let years = n_periods / self.config.periods_per_year;
        let annualized_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        let mean_return = portfolio_returns.iter().sum::<f64>() / n_periods;
        let std_return = {
            let variance = portfolio_returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / n_periods;
            variance.sqrt()
        };

        let rf_per_period = self.config.risk_free_rate / self.config.periods_per_year;
        let sharpe_ratio = if std_return > 1e-10 {
            ((mean_return - rf_per_period) / std_return) * self.config.periods_per_year.sqrt()
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = portfolio_returns
            .iter()
            .filter(|&&r| r < rf_per_period)
            .map(|&r| (r - rf_per_period).powi(2))
            .collect();
        let downside_dev = if !downside_returns.is_empty() {
            (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt()
        } else {
            1e-10
        };
        let sortino_ratio =
            ((mean_return - rf_per_period) / downside_dev) * self.config.periods_per_year.sqrt();

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let profit_factor = if gross_loss > 1e-10 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let calmar_ratio = if max_drawdown > 1e-10 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        let result = BacktestResult {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            total_trades,
            profit_factor,
            calmar_ratio,
            final_value: equity,
            equity_curve,
            returns: portfolio_returns,
        };

        debug!(
            "Backtest complete: return={:.2}%, sharpe={:.2}, max_dd={:.2}%",
            result.total_return * 100.0,
            result.sharpe_ratio,
            result.max_drawdown * 100.0
        );

        Ok(result)
    }
}

impl BacktestResult {
    /// Print a formatted summary of the backtest results.
    pub fn print_summary(&self) {
        println!("=== Backtest Results ===");
        println!("Total Return:      {:.2}%", self.total_return * 100.0);
        println!(
            "Annualized Return: {:.2}%",
            self.annualized_return * 100.0
        );
        println!("Sharpe Ratio:      {:.4}", self.sharpe_ratio);
        println!("Sortino Ratio:     {:.4}", self.sortino_ratio);
        println!("Max Drawdown:      {:.2}%", self.max_drawdown * 100.0);
        println!("Win Rate:          {:.2}%", self.win_rate * 100.0);
        println!("Total Trades:      {}", self.total_trades);
        println!("Profit Factor:     {:.4}", self.profit_factor);
        println!("Calmar Ratio:      {:.4}", self.calmar_ratio);
        println!("Final Value:       ${:.2}", self.final_value);
        println!("========================");
    }
}
