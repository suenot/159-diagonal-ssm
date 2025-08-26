//! Bybit API client implementation.
//!
//! Fetches kline (candlestick) data from the Bybit v5 REST API.

use anyhow::Result;
use log::{debug, info};
use reqwest::Client;

use super::types::{BybitResponse, Kline};

/// Base URL for the Bybit v5 market API.
const BYBIT_API_BASE: &str = "https://api.bybit.com/v5/market/kline";

/// Client for interacting with the Bybit market data API.
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Creates a new BybitClient with default settings.
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Creates a new BybitClient with a custom base URL (useful for testing).
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline data for a given symbol and interval.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol, e.g. "BTCUSDT"
    /// * `interval` - Kline interval: "1", "5", "15", "60", "240", "D", "W"
    /// * `limit` - Number of klines to retrieve (max 200)
    ///
    /// # Returns
    ///
    /// A vector of [`Kline`] records sorted by timestamp ascending.
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let limit = limit.min(200);
        info!(
            "Fetching {} klines for {} at {} interval",
            limit, symbol, interval
        );

        let response = self
            .client
            .get(&self.base_url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let status = response.status();
        debug!("API response status: {}", status);

        if !status.is_success() {
            let body = response.text().await?;
            anyhow::bail!("API request failed with status {}: {}", status, body);
        }

        let bybit_response: BybitResponse = response.json().await?;

        if bybit_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error (code {}): {}",
                bybit_response.ret_code,
                bybit_response.ret_msg
            );
        }

        let mut klines: Vec<Kline> = bybit_response
            .result
            .list
            .iter()
            .map(|raw| Kline::from_raw(raw))
            .collect::<Result<Vec<_>>>()?;

        // Bybit returns newest first; reverse to get chronological order
        klines.reverse();

        info!("Successfully fetched {} klines", klines.len());
        Ok(klines)
    }

    /// Fetch klines with a start and end time range.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Trading pair symbol
    /// * `interval` - Kline interval
    /// * `start` - Start timestamp in milliseconds
    /// * `end` - End timestamp in milliseconds
    /// * `limit` - Maximum number of klines
    pub async fn fetch_klines_range(
        &self,
        symbol: &str,
        interval: &str,
        start: u64,
        end: u64,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let limit = limit.min(200);
        info!(
            "Fetching klines for {} from {} to {} at {} interval",
            symbol, start, end, interval
        );

        let response = self
            .client
            .get(&self.base_url)
            .query(&[
                ("category", "spot"),
                ("symbol", symbol),
                ("interval", interval),
                ("start", &start.to_string()),
                ("end", &end.to_string()),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        let bybit_response: BybitResponse = response.json().await?;

        if bybit_response.ret_code != 0 {
            anyhow::bail!(
                "Bybit API error (code {}): {}",
                bybit_response.ret_code,
                bybit_response.ret_msg
            );
        }

        let mut klines: Vec<Kline> = bybit_response
            .result
            .list
            .iter()
            .map(|raw| Kline::from_raw(raw))
            .collect::<Result<Vec<_>>>()?;

        klines.reverse();
        Ok(klines)
    }
}
