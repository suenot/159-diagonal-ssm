//! Example: Train a Diagonal SSM model on synthetic data.
//!
//! Creates synthetic price data with a trend component, builds a dataset,
//! trains the model, and reports training loss.
//!
//! Usage:
//!   cargo run --example train

use diagonal_ssm::{DataLoader, DiagonalSSMConfig, DiagonalSSMModel};
use rand::Rng;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== Diagonal SSM Training Example ===\n");

    // Generate synthetic price data with trend + noise
    let mut rng = rand::thread_rng();
    let n_points = 500;
    let mut prices = Vec::with_capacity(n_points);
    let mut price = 100.0;

    for i in 0..n_points {
        // Sine trend + random noise
        let trend = 0.01 * (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin();
        let noise = 0.005 * rng.gen_range(-1.0..1.0);
        price *= 1.0 + trend + noise;
        prices.push(price);
    }

    println!("Generated {} synthetic prices", n_points);
    println!(
        "Price range: {:.2} - {:.2}",
        prices.iter().cloned().fold(f64::INFINITY, f64::min),
        prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Build dataset
    let loader = DataLoader::new().seq_len(30).target_horizon(1);
    let mut dataset = loader.build_from_prices(&prices)?;
    println!(
        "Dataset: {} windows, {} features",
        dataset.n_windows, dataset.n_features
    );

    // Normalize features
    let (means, stds) = dataset.normalize();
    println!("Feature means: {:?}", means);
    println!("Feature stds:  {:?}", stds);

    // Create model
    let config = DiagonalSSMConfig::s4d_lin(16, dataset.n_features)
        .with_learning_rate(0.01)
        .with_epochs(50)
        .with_num_layers(1);

    println!("\nModel config:");
    println!("  State dim:     {}", config.state_dim);
    println!("  D model:       {}", config.d_model);
    println!("  Layers:        {}", config.num_layers);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Init method:   {:?}", config.init_method);

    let mut model = DiagonalSSMModel::new(config);

    // Train
    println!("\nTraining...");
    let losses = model.train(&dataset)?;

    // Print loss progression
    println!("\nLoss progression:");
    for (i, loss) in losses.iter().enumerate() {
        if i % 10 == 0 || i == losses.len() - 1 {
            println!("  Epoch {:3}: loss = {:.6}", i + 1, loss);
        }
    }

    // Show eigenvalue spectrum
    if let Some(eigenvalues) = model.eigenvalues() {
        println!("\nEigenvalue spectrum (first 5):");
        for (i, ev) in eigenvalues.iter().take(5).enumerate() {
            println!(
                "  λ_{} = {:.4} + {:.4}j  (|λ| = {:.4})",
                i,
                ev.re,
                ev.im,
                ev.norm()
            );
        }
    }

    println!("\nTraining complete!");
    Ok(())
}
