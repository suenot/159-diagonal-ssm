//! Example: Make predictions with a Diagonal SSM model.
//!
//! Creates a model and runs inference on synthetic data,
//! demonstrating both convolution and recurrent prediction modes.
//!
//! Usage:
//!   cargo run --example predict

use diagonal_ssm::{DataLoader, DiagonalSSMConfig, DiagonalSSMModel, SignalGenerator};
use rand::Rng;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== Diagonal SSM Prediction Example ===\n");

    // Generate synthetic data
    let mut rng = rand::thread_rng();
    let n_points = 300;
    let mut prices = Vec::with_capacity(n_points);
    let mut price = 50000.0; // Simulating BTC-like prices

    for _ in 0..n_points {
        let noise = 0.002 * rng.gen_range(-1.0..1.0);
        price *= 1.0 + noise;
        prices.push(price);
    }

    // Build dataset
    let loader = DataLoader::new().seq_len(20).target_horizon(1);
    let mut dataset = loader.build_from_prices(&prices)?;
    dataset.normalize();

    println!("Dataset: {} windows", dataset.n_windows);

    // Create model (untrained, just for demonstration)
    let config = DiagonalSSMConfig::s4d_lin(8, dataset.n_features).with_num_layers(1);
    let model = DiagonalSSMModel::new(config);

    // Generate predictions
    println!("\nGenerating predictions...");
    let predictions = model.predict(&dataset);

    // Show prediction statistics
    let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;
    let min_pred = predictions.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_pred = predictions
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("\n=== Prediction Statistics ===");
    println!("Count:   {}", predictions.len());
    println!("Mean:    {:.4}", mean_pred);
    println!("Min:     {:.4}", min_pred);
    println!("Max:     {:.4}", max_pred);

    // Generate trading signals
    let signal_gen = SignalGenerator::default();
    let signals = signal_gen.generate_all(&predictions);

    let n_long = signals
        .iter()
        .filter(|&&s| s == diagonal_ssm::strategy::signals::Signal::Long)
        .count();
    let n_short = signals
        .iter()
        .filter(|&&s| s == diagonal_ssm::strategy::signals::Signal::Short)
        .count();
    let n_flat = signals
        .iter()
        .filter(|&&s| s == diagonal_ssm::strategy::signals::Signal::Flat)
        .count();

    println!("\n=== Signal Distribution ===");
    println!("Long:  {} ({:.1}%)", n_long, 100.0 * n_long as f64 / signals.len() as f64);
    println!("Short: {} ({:.1}%)", n_short, 100.0 * n_short as f64 / signals.len() as f64);
    println!("Flat:  {} ({:.1}%)", n_flat, 100.0 * n_flat as f64 / signals.len() as f64);

    // Show first 10 predictions with signals
    println!("\n=== Sample Predictions ===");
    println!("{:<6} {:<10} {:<8}", "Index", "Prob", "Signal");
    for (i, (&pred, &signal)) in predictions.iter().zip(signals.iter()).take(10).enumerate() {
        let signal_str = match signal {
            diagonal_ssm::strategy::signals::Signal::Long => "LONG",
            diagonal_ssm::strategy::signals::Signal::Short => "SHORT",
            diagonal_ssm::strategy::signals::Signal::Flat => "FLAT",
        };
        println!("{:<6} {:<10.4} {:<8}", i, pred, signal_str);
    }

    // Compare convolution vs recurrent mode
    println!("\n=== Convolution vs Recurrent Mode ===");
    let sample_input = dataset.get_window(0);
    let conv_pred = model.forward(&sample_input);
    let rec_pred = model.forward_recurrent(&sample_input);
    println!("Convolution mode: {:.6}", conv_pred);
    println!("Recurrent mode:   {:.6}", rec_pred);
    println!("Difference:       {:.2e}", (conv_pred - rec_pred).abs());

    Ok(())
}
