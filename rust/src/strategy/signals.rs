//! Trading signal generation from model predictions.

use log::debug;

/// Trading signal direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Go long (buy).
    Long,
    /// Go short (sell).
    Short,
    /// Stay flat (no position).
    Flat,
}

/// Generates trading signals from model prediction probabilities.
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    /// Threshold above which to go long (probability > long_threshold).
    pub long_threshold: f64,
    /// Threshold below which to go short (probability < short_threshold).
    pub short_threshold: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self {
            long_threshold: 0.55,
            short_threshold: 0.45,
        }
    }
}

impl SignalGenerator {
    /// Create a new signal generator with custom thresholds.
    pub fn new(long_threshold: f64, short_threshold: f64) -> Self {
        Self {
            long_threshold,
            short_threshold,
        }
    }

    /// Generate a signal from a single prediction probability.
    ///
    /// - probability > long_threshold => Long
    /// - probability < short_threshold => Short
    /// - otherwise => Flat
    pub fn generate(&self, probability: f64) -> Signal {
        if probability > self.long_threshold {
            Signal::Long
        } else if probability < self.short_threshold {
            Signal::Short
        } else {
            Signal::Flat
        }
    }

    /// Generate signals from a vector of predictions.
    pub fn generate_all(&self, predictions: &[f64]) -> Vec<Signal> {
        let signals: Vec<Signal> = predictions.iter().map(|&p| self.generate(p)).collect();

        let n_long = signals.iter().filter(|&&s| s == Signal::Long).count();
        let n_short = signals.iter().filter(|&&s| s == Signal::Short).count();
        let n_flat = signals.iter().filter(|&&s| s == Signal::Flat).count();

        debug!(
            "Generated {} signals: {} long, {} short, {} flat",
            signals.len(),
            n_long,
            n_short,
            n_flat
        );

        signals
    }
}
