"""
Example usage of the Diagonal SSM for financial time series prediction.

This script demonstrates the full pipeline:
    1. Generate synthetic stock data (no external API needed).
    2. Engineer features (returns, volatility, RSI, MACD, volume_change).
    3. Build a DiagonalSSMForecaster model.
    4. Train the model for a few epochs.
    5. Generate predictions on held-out data.
    6. Run a backtest and print performance metrics.

Run from the chapter directory:
    python -m python.example_usage
Or directly:
    python python/example_usage.py
"""

import sys
import os

# Ensure the parent directory is on the path so imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from python.model import DiagonalSSMConfig, DiagonalSSMForecaster
from python.data import StockDataLoader, engineer_features, DiagonalSSMDataset
from python.strategy import Backtester, BacktestConfig


def main():
    # ------------------------------------------------------------------
    # 0. Settings
    # ------------------------------------------------------------------
    # Uncomment the next line to enable debug logging:
    # logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    SEQ_LEN = 64
    PREDICTION_HORIZON = 8
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-3

    # ------------------------------------------------------------------
    # 1. Generate synthetic data
    # ------------------------------------------------------------------
    print("\n--- Step 1: Generating synthetic stock data ---")
    loader = StockDataLoader()
    raw_df = loader.generate(num_bars=3000, seed=SEED)
    print(f"Generated {len(raw_df)} bars of synthetic OHLCV data")
    print(raw_df.head())

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("\n--- Step 2: Engineering features ---")
    features_df = engineer_features(raw_df)
    print(f"Features shape: {features_df.shape}")
    print(f"Feature columns: {list(features_df.columns)}")

    # ------------------------------------------------------------------
    # 3. Create dataset and dataloaders
    # ------------------------------------------------------------------
    print("\n--- Step 3: Preparing datasets ---")
    n_total = len(features_df)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    train_features = features_df.iloc[:n_train]
    val_features = features_df.iloc[n_train:n_train + n_val]
    test_features = features_df.iloc[n_train + n_val:]

    close_prices = raw_df["close"].loc[features_df.index]

    train_ds = DiagonalSSMDataset(
        train_features, close_prices, seq_len=SEQ_LEN, prediction_horizon=PREDICTION_HORIZON,
    )
    val_ds = DiagonalSSMDataset(
        val_features, close_prices, seq_len=SEQ_LEN, prediction_horizon=PREDICTION_HORIZON,
    )
    test_ds = DiagonalSSMDataset(
        test_features, close_prices, seq_len=SEQ_LEN, prediction_horizon=PREDICTION_HORIZON,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples | Test: {len(test_ds)} samples")

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    print("\n--- Step 4: Building DiagonalSSMForecaster ---")
    config = DiagonalSSMConfig(
        input_features=len(features_df.columns),
        state_dim=32,
        d_model=64,
        num_layers=2,
        dropout=0.1,
        init_method="s4d_lin",
        bidirectional=False,
        prediction_horizon=PREDICTION_HORIZON,
        lr_dt=True,
        seq_len=SEQ_LEN,
    )
    model = DiagonalSSMForecaster(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print(f"\n--- Step 5: Training for {NUM_EPOCHS} epochs ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.MSELoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                val_loss += criterion(preds, y_batch).item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        scheduler.step()

        print(f"  Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # ------------------------------------------------------------------
    # 6. Generate predictions on test set
    # ------------------------------------------------------------------
    print("\n--- Step 6: Generating test predictions ---")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    print(f"Predictions shape: {predictions.shape}")

    # Simple regression metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    print(f"Test MSE: {mse:.8f} | Test MAE: {mae:.8f}")

    # Direction accuracy (sign of mean predicted return vs actual)
    pred_dir = np.sign(predictions.mean(axis=1))
    true_dir = np.sign(targets.mean(axis=1))
    dir_accuracy = np.mean(pred_dir == true_dir)
    print(f"Direction accuracy: {dir_accuracy:.4f}")

    # ------------------------------------------------------------------
    # 7. Backtest
    # ------------------------------------------------------------------
    print("\n--- Step 7: Running backtest ---")
    # Align timestamps: test predictions start at index (seq_len) of the test features
    test_start_idx = SEQ_LEN
    test_timestamps = test_features.index[test_start_idx: test_start_idx + len(predictions)]

    bt_config = BacktestConfig(
        initial_capital=100_000.0,
        commission=0.001,
        slippage=0.0005,
        buy_threshold=0.0005,
        sell_threshold=-0.0005,
        stop_loss=0.03,
    )
    backtester = Backtester(bt_config)
    signals = backtester.generate_signals(predictions, test_timestamps)
    result = backtester.run(signals, close_prices)

    print(result.summary())

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("--- Done ---")
    print(f"Diagonal SSM ({config.num_layers} layers, d_model={config.d_model}, state_dim={config.state_dim})")
    print(f"Init method: {config.init_method} | Bidirectional: {config.bidirectional}")
    print(f"Total parameters: {model.count_parameters():,}")


if __name__ == "__main__":
    main()
