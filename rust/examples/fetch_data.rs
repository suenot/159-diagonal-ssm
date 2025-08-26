//! Example: Fetch kline data from the Bybit API.
//!
//! Demonstrates how to use the BybitClient to fetch candlestick data
//! and print a summary of the retrieved data.
//!
//! Usage:
//!   cargo run --example fetch_data

use diagonal_ssm::BybitClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let client = BybitClient::new();

    println!("Fetching BTCUSDT klines from Bybit API...");
    let klines = client.fetch_klines("BTCUSDT", "60", 100).await?;

    println!("\n=== Data Summary ===");
    println!("Symbol:     BTCUSDT");
    println!("Interval:   1 hour");
    println!("Records:    {}", klines.len());

    if let Some(first) = klines.first() {
        println!("First time: {}", first.timestamp);
        println!("First close: ${:.2}", first.close);
    }

    if let Some(last) = klines.last() {
        println!("Last time:  {}", last.timestamp);
        println!("Last close: ${:.2}", last.close);
    }

    // Price statistics
    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_price = closes.iter().sum::<f64>() / closes.len() as f64;

    println!("\n=== Price Statistics ===");
    println!("Min:     ${:.2}", min_price);
    println!("Max:     ${:.2}", max_price);
    println!("Average: ${:.2}", avg_price);

    // Volume statistics
    let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
    let total_volume: f64 = volumes.iter().sum();
    let avg_volume = total_volume / volumes.len() as f64;

    println!("\n=== Volume Statistics ===");
    println!("Total volume: {:.2}", total_volume);
    println!("Avg volume:   {:.2}", avg_volume);

    println!("\n=== Sample Klines (first 5) ===");
    for (i, k) in klines.iter().take(5).enumerate() {
        println!(
            "  [{}] ts={} O={:.2} H={:.2} L={:.2} C={:.2} V={:.2}",
            i, k.timestamp, k.open, k.high, k.low, k.close, k.volume
        );
    }

    Ok(())
}
