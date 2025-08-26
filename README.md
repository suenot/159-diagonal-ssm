# Chapter 138: Diagonal State Space Models for Trading

This chapter explores **Diagonal State Space Models (Diagonal SSMs)**, a simplified yet highly effective variant of Structured State Space Models (S4) that achieves comparable performance with significantly reduced computational complexity. By constraining the state matrix **A** to be diagonal, these models enable efficient parallel training and fast inference, making them ideal for financial time series prediction.

## Contents

1. [Introduction to Diagonal SSMs](#introduction-to-diagonal-ssms)
    * [From S4 to Diagonal SSMs](#from-s4-to-diagonal-ssms)
    * [Key Advantages](#key-advantages)
    * [Comparison with Other SSM Variants](#comparison-with-other-ssm-variants)
2. [Mathematical Foundation](#mathematical-foundation)
    * [State Space Model Basics](#state-space-model-basics)
    * [Diagonal Parameterization](#diagonal-parameterization)
    * [Discretization](#discretization)
    * [Efficient Convolution](#efficient-convolution)
3. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: Diagonal SSM Architecture](#02-diagonal-ssm-architecture)
    * [03: Model Training](#03-model-training)
    * [04: Financial Time Series Prediction](#04-financial-time-series-prediction)
    * [05: Backtesting Strategy](#05-backtesting-strategy)
4. [Rust Implementation](#rust-implementation)
5. [Python Implementation](#python-implementation)
6. [Best Practices](#best-practices)
7. [Resources](#resources)

## Introduction to Diagonal SSMs

### From S4 to Diagonal SSMs

The Structured State Space Sequence model (S4) introduced the HiPPO-initialized state matrix for capturing long-range dependencies in sequences. However, S4 requires sophisticated matrix decomposition techniques (e.g., Normal Plus Low-Rank, or NPLR) to handle the dense state matrix **A**, resulting in complex implementation and training procedures.

**Diagonal State Spaces** (DSS), introduced by Gupta et al. (2022) in the paper "Diagonal State Spaces are as Effective as Structured State Spaces," demonstrated that simply restricting the state matrix to be diagonal achieves performance on par with S4, while dramatically simplifying both implementation and computation.

```
S4 Architecture (Complex):
┌─────────────────────────────────────────────────┐
│  x(t) = A·x(t-1) + B·u(t)                     │
│  y(t) = C·x(t) + D·u(t)                        │
│                                                  │
│  A ∈ ℂ^{N×N}  (Dense, requires NPLR decomp.)   │
│  Eigenvalue decomposition needed                 │
│  Complex Cauchy kernel computation               │
└─────────────────────────────────────────────────┘

Diagonal SSM (Simple):
┌─────────────────────────────────────────────────┐
│  x(t) = Λ·x(t-1) + B·u(t)                     │
│  y(t) = C·x(t) + D·u(t)                        │
│                                                  │
│  Λ = diag(λ₁, λ₂, ..., λₙ) ∈ ℂ^N             │
│  Each state evolves independently!               │
│  Trivially parallelizable                        │
└─────────────────────────────────────────────────┘
```

### Key Advantages

1. **Simplified Implementation**
   - No need for NPLR decomposition
   - Diagonal structure is trivially diagonalizable
   - Standard autodiff works out of the box

2. **Computational Efficiency**
   - O(N) per state update instead of O(N²)
   - Embarrassingly parallel across state dimensions
   - Efficient convolution via FFT

3. **Comparable Performance**
   - Matches S4 on Long Range Arena benchmarks
   - Strong results on sequential CIFAR, speech, text
   - Effective for financial time series

4. **Training Stability**
   - Simpler optimization landscape
   - Easier initialization strategies
   - More robust gradient flow

### Comparison with Other SSM Variants

| Model | State Matrix | Complexity | Parallelizable | Implementation |
|-------|-------------|------------|----------------|----------------|
| **Diagonal SSM** | Diagonal Λ | O(NL) | Yes (trivial) | Simple |
| S4 | NPLR A | O(NL log L) | Yes (complex) | Complex |
| S4D | Diagonal (S4 variant) | O(NL) | Yes | Moderate |
| Mamba | Data-dependent | O(NL) | Yes (scan) | Moderate |
| Linear RNN | Dense A | O(N²L) | No | Simple |
| HiPPO-RNN | HiPPO A | O(N²L) | No | Moderate |

## Mathematical Foundation

### State Space Model Basics

A continuous-time state space model maps an input signal u(t) to an output y(t) through a hidden state x(t):

```
dx/dt = A·x(t) + B·u(t)
  y(t) = C·x(t) + D·u(t)
```

Where:
- **A** ∈ ℝ^{N×N}: State transition matrix
- **B** ∈ ℝ^{N×1}: Input projection
- **C** ∈ ℝ^{1×N}: Output projection
- **D** ∈ ℝ: Skip connection (feedthrough)
- **N**: State dimension (hidden size)

### Diagonal Parameterization

In Diagonal SSMs, we constrain A to be diagonal:

```
A = Λ = diag(λ₁, λ₂, ..., λₙ)
```

This means each state dimension evolves independently:

```
dxᵢ/dt = λᵢ·xᵢ(t) + Bᵢ·u(t)    for i = 1, ..., N
```

The eigenvalues λᵢ are complex-valued to allow oscillatory behavior:

```
λᵢ = aᵢ + j·bᵢ    where aᵢ < 0 (stability constraint)
```

**Initialization**: Following DSS, the diagonal elements can be initialized using:
1. **HiPPO-inspired**: Approximate the HiPPO matrix eigenvalues
2. **Log-uniform**: λᵢ = -exp(uniform(log(0.001), log(0.1))) + j·uniform(0, π)
3. **S4D-Lin**: λₙ = -1/2 + j·π·n for uniform spacing

### Discretization

To apply the continuous model to discrete sequences with step size Δ, we discretize using the Zero-Order Hold (ZOH) method:

```
Ā = exp(Λ·Δ)           (element-wise for diagonal)
B̄ = (Ā - I) · Λ⁻¹ · B  (simplified for diagonal)

Discrete recurrence:
x[k] = Ā·x[k-1] + B̄·u[k]
y[k] = C·x[k] + D·u[k]
```

For diagonal Λ, the matrix exponential is simply element-wise:

```
Ā = diag(exp(λ₁·Δ), exp(λ₂·Δ), ..., exp(λₙ·Δ))
```

### Efficient Convolution

The SSM can be computed as a convolution with kernel K:

```
K = (C·B̄, C·Ā·B̄, C·Ā²·B̄, ..., C·Ā^{L-1}·B̄)

y = K * u    (convolution, computed via FFT in O(L log L))
```

For diagonal Ā, each element of K is:

```
K[l] = Σᵢ Cᵢ · Āᵢˡ · B̄ᵢ    for l = 0, ..., L-1
```

This can be computed efficiently using the Vandermonde product:

```
K = C ⊙ B̄ · V(Ā, L)

where V(Ā, L) = [1, Ā, Ā², ..., Ā^{L-1}]  (element-wise powers)
```

## Practical Examples

### 01: Data Preparation

We use both stock market data (via Yahoo Finance) and cryptocurrency data (via Bybit API).

```python
from python.data import BybitClient, StockDataLoader, DiagonalSSMDataset

# Fetch crypto data from Bybit
client = BybitClient()
btc_klines = client.get_klines("BTCUSDT", interval="60", limit=1000)
eth_klines = client.get_klines("ETHUSDT", interval="60", limit=1000)

# Fetch stock data
stock_loader = StockDataLoader()
spy_data = stock_loader.get_stock_data("SPY", period="2y")

# Create dataset with features
dataset = DiagonalSSMDataset(
    data=btc_klines,
    seq_len=168,        # 1 week of hourly data
    pred_horizon=24,    # Predict 24 hours ahead
    features=["returns", "volatility", "volume_change", "rsi", "macd"]
)
```

### 02: Diagonal SSM Architecture

```python
from python.model import DiagonalSSMConfig, DiagonalSSMForecaster

config = DiagonalSSMConfig(
    input_features=6,
    state_dim=64,
    d_model=128,
    num_layers=4,
    dropout=0.1,
    init_method="s4d_lin",   # S4D-Lin initialization
    bidirectional=False,
    prediction_horizon=24
)

model = DiagonalSSMForecaster(config)
```

### 03: Model Training

```python
import torch
from python.model import DiagonalSSMForecaster, DiagonalSSMConfig

config = DiagonalSSMConfig(state_dim=64, d_model=128, num_layers=4)
model = DiagonalSSMForecaster(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    for batch in train_loader:
        x, y = batch
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
```

### 04: Financial Time Series Prediction

The Diagonal SSM processes financial time series by treating each feature (price, volume, indicators) as an input channel, passing it through multiple diagonal SSM layers that capture temporal dependencies at different scales.

```
Input Features          Diagonal SSM Layers          Prediction
┌──────────┐    ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ Returns   │───>│        │─>│        │─>│        │─>│        │
│ Volume    │───>│ Layer  │─>│ Layer  │─>│ Layer  │─>│ Return │
│ RSI       │───>│   1    │─>│   2    │─>│   3    │─>│ Pred.  │
│ MACD      │───>│        │─>│        │─>│        │─>│        │
│ Volatility│───>│        │─>│        │─>│        │─>│        │
└──────────┘    └────────┘  └────────┘  └────────┘  └────────┘
                 (State=64)  (State=64)  (State=64)
                 Diagonal Λ  Diagonal Λ  Diagonal Λ
```

### 05: Backtesting Strategy

```python
from python.strategy import BacktestConfig, Backtester

config = BacktestConfig(
    initial_capital=100000.0,
    commission=0.001,
    slippage=0.0005,
    long_threshold=0.001,
    short_threshold=-0.001,
    stop_loss_level=0.05
)

backtester = Backtester(config)
result = backtester.run(model, test_dataset)

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Sortino Ratio: {result.sortino_ratio:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

## Rust Implementation

The Rust implementation provides a high-performance version of the Diagonal SSM model with Bybit API integration:

```rust
use diagonal_ssm::{BybitClient, DataLoader, DiagonalSSMModel, DiagonalSSMConfig};

#[tokio::main]
async fn main() {
    // Fetch data from Bybit
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", "60", 1000).await.unwrap();

    // Prepare dataset
    let loader = DataLoader::new()
        .seq_len(168)
        .target_horizon(24);
    let dataset = loader.prepare_dataset(&klines).unwrap();

    // Create and train model
    let config = DiagonalSSMConfig {
        state_dim: 64,
        d_model: 128,
        num_layers: 4,
        ..Default::default()
    };
    let mut model = DiagonalSSMModel::new(config);
    model.train(&dataset, 100, 1e-3).unwrap();

    // Run backtest
    let results = model.backtest(&dataset).unwrap();
    println!("Sharpe: {:.3}", results.sharpe_ratio);
}
```

See `rust/` directory for the full implementation.

## Python Implementation

The Python implementation uses PyTorch for the model and provides:
- `python/model.py`: Diagonal SSM layer and forecaster model
- `python/data.py`: Data loading from Bybit and Yahoo Finance
- `python/strategy.py`: Backtesting framework with risk management
- `python/example_usage.py`: Complete usage example

See `python/` directory for full source code.

## Best Practices

1. **Initialization Matters**: Use S4D-Lin or HiPPO-inspired initialization for the diagonal elements. Random initialization often leads to poor convergence.

2. **Complex vs Real**: Using complex-valued diagonal elements allows the model to capture oscillatory patterns (common in financial markets). Ensure your implementation handles complex arithmetic.

3. **Step Size Δ**: The discretization step size is a learnable parameter per layer. Initialize it with log-uniform distribution in [0.001, 0.1].

4. **Stability Constraint**: Keep the real parts of diagonal eigenvalues negative (Re(λᵢ) < 0) to ensure stability. Apply a softplus or exponential reparameterization.

5. **Normalization**: Apply layer normalization after each SSM layer. This stabilizes training and improves generalization.

6. **Financial Data**: Normalize returns and features to zero mean and unit variance. Use rolling statistics to avoid lookahead bias.

7. **Sequence Length**: Diagonal SSMs handle long sequences efficiently. Use longer lookback windows (168-720 hours) for better pattern capture.

## Resources

1. **Diagonal State Spaces are as Effective as Structured State Spaces** — Gupta, Hasani, Sontag (2022)
   - URL: https://arxiv.org/abs/2203.14343

2. **Efficiently Modeling Long Sequences with Structured State Spaces (S4)** — Gu, Goel, Ré (2022)
   - URL: https://arxiv.org/abs/2111.00396

3. **On the Parameterization and Initialization of Diagonal State Space Models (S4D)** — Gu, Gupta, Goel, Ré (2022)
   - URL: https://arxiv.org/abs/2206.11893

4. **How to Train Your HiPPO** — Gu, Johnson, Timeseries, Dao, Rudra, Ré (2022)
   - URL: https://arxiv.org/abs/2206.12037
