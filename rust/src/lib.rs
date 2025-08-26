//! # Diagonal SSM (S4D) for Trading
//!
//! Implementation of Diagonal State Space Models based on the S4D architecture
//! (Structured State Spaces for Sequence Modeling with Diagonal Approximation)
//! applied to financial time series prediction and trading signal generation.
//!
//! ## Architecture
//!
//! The model uses a diagonal approximation of the state space:
//!
//! ```text
//! x'(t) = Λ x(t) + B u(t)
//! y(t)  = C x(t) + D u(t)
//! ```
//!
//! where Λ is a diagonal matrix of complex eigenvalues initialized via S4D-Lin:
//! `λₙ = -0.5 + j·π·n` for n = 0, 1, ..., N-1.
//!
//! ## Modules
//!
//! - [`api`] - Bybit API client for fetching market data
//! - [`data`] - Data loading, feature engineering, and dataset construction
//! - [`model`] - Diagonal SSM model with S4D initialization and training
//! - [`strategy`] - Trading signal generation and backtesting

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::Kline;
pub use data::dataset::Dataset;
pub use data::features::FeatureEngine;
pub use data::loader::DataLoader;
pub use model::config::DiagonalSSMConfig;
pub use model::diagonal_ssm::DiagonalSSMModel;
pub use strategy::backtest::{BacktestConfig, BacktestResult, Backtester};
pub use strategy::signals::SignalGenerator;
