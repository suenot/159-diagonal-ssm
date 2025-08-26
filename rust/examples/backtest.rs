//! Example: Full backtest pipeline with Diagonal SSM.
//!
//! Generates synthetic data, trains a model, generates signals,
//! and runs a backtest to evaluate trading performance.
//!
//! Usage:
//!   cargo run --example backtest

use diagonal_ssm::{
    BacktestConfig, Backtester, DataLoader, DiagonalSSMConfig, DiagonalSSMModel, SignalGenerator,
};
use rand::Rng;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== Diagonal SSM Backtest Example ===\n");

    // 1. Generate synthetic price data
    let mut rng = rand::thread_rng();
    let n_points = 600;
    let mut prices = Vec::with_capacity(n_points);
    let mut price = 100.0;
    let mut returns_raw = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let trend = 0.005 * (2.0 * std::f64::consts::PI * i as f64 / 120.0).sin();
        let noise = 0.008 * rng.gen_range(-1.0..1.0);
        let ret = trend + noise;
        price *= 1.0 + ret;
        prices.push(price);
        returns_raw.push(ret);
    }

    println!("Generated {} synthetic prices", n_points);

    // 2. Build dataset
    let seq_len = 30;
    let loader = DataLoader::new().seq_len(seq_len).target_horizon(1);
    let mut dataset = loader.build_from_prices(&prices)?;
    dataset.normalize();

    // 3. Split into train/test
    let (train_data, test_data) = dataset.train_val_split(0.7);
    println!(
        "Train: {} windows, Test: {} windows",
        train_data.n_windows, test_data.n_windows
    );

    // 4. Create and train model
    let config = DiagonalSSMConfig::s4d_lin(16, train_data.n_features)
        .with_learning_rate(0.005)
        .with_epochs(30)
        .with_num_layers(1);

    let mut model = DiagonalSSMModel::new(config);

    println!("\nTraining model...");
    let losses = model.train(&train_data)?;
    println!(
        "Final training loss: {:.6}",
        losses.last().unwrap_or(&0.0)
    );

    // 5. Generate predictions on test set
    println!("\nGenerating predictions on test set...");
    let predictions = model.predict(&test_data);

    // 6. Generate trading signals
    let signal_gen = SignalGenerator::new(0.52, 0.48);
    let signals = signal_gen.generate_all(&predictions);

    // 7. Extract corresponding returns for backtesting
    // Use a portion of the raw returns aligned with the test windows
    let warmup = 26; // Feature warmup period (MACD long window)
    let test_start = warmup + seq_len + (train_data.n_windows);
    let test_returns: Vec<f64> = returns_raw
        .iter()
        .skip(test_start)
        .take(test_data.n_windows)
        .copied()
        .collect();

    // Ensure we have enough returns (pad with zeros if needed)
    let test_returns: Vec<f64> = if test_returns.len() < signals.len() {
        let mut padded = test_returns;
        padded.resize(signals.len(), 0.0);
        padded
    } else {
        test_returns[..signals.len()].to_vec()
    };

    // 8. Run backtest
    println!("\nRunning backtest...");
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        transaction_cost: 0.001,
        position_size: 1.0,
        risk_free_rate: 0.02,
        periods_per_year: 252.0,
    };

    let backtester = Backtester::new(backtest_config);
    let result = backtester.run(&signals, &test_returns)?;

    // 9. Print results
    println!();
    result.print_summary();

    // 10. Print equity curve summary
    let eq = &result.equity_curve;
    if eq.len() > 1 {
        println!("\n=== Equity Curve (sampled) ===");
        let step = eq.len() / 10;
        for i in (0..eq.len()).step_by(step.max(1)) {
            println!("  Step {:4}: ${:.2}", i, eq[i]);
        }
        println!("  Step {:4}: ${:.2} (final)", eq.len() - 1, eq[eq.len() - 1]);
    }

    Ok(())
}
