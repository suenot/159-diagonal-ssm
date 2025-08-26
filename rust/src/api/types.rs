//! API response types for Bybit market data.

use serde::{Deserialize, Serialize};

/// A single kline (candlestick) record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp in milliseconds.
    pub timestamp: u64,
    /// Opening price.
    pub open: f64,
    /// Highest price.
    pub high: f64,
    /// Lowest price.
    pub low: f64,
    /// Closing price.
    pub close: f64,
    /// Trading volume.
    pub volume: f64,
    /// Turnover (quote volume).
    pub turnover: f64,
}

/// Raw kline data as returned by the Bybit API (array of strings).
#[derive(Debug, Clone, Deserialize)]
pub struct KlineData {
    /// Trading pair symbol.
    pub symbol: String,
    /// Kline interval category.
    pub category: String,
    /// List of kline entries, each as a Vec of strings.
    pub list: Vec<Vec<String>>,
}

/// Top-level Bybit API response wrapper.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BybitResponse {
    /// Return code (0 = success).
    pub ret_code: i32,
    /// Return message.
    pub ret_msg: String,
    /// Response result payload.
    pub result: KlineData,
}

impl Kline {
    /// Parse a kline from the raw string array returned by Bybit.
    ///
    /// Expected format: [timestamp, open, high, low, close, volume, turnover]
    pub fn from_raw(raw: &[String]) -> anyhow::Result<Self> {
        if raw.len() < 7 {
            anyhow::bail!(
                "Expected at least 7 fields in kline data, got {}",
                raw.len()
            );
        }
        Ok(Self {
            timestamp: raw[0].parse()?,
            open: raw[1].parse()?,
            high: raw[2].parse()?,
            low: raw[3].parse()?,
            close: raw[4].parse()?,
            volume: raw[5].parse()?,
            turnover: raw[6].parse()?,
        })
    }

    /// Returns the mid price (average of high and low).
    pub fn mid_price(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    /// Returns the typical price (average of high, low, close).
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}
