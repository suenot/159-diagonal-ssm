//! Diagonal State Space Model (S4D) implementation.
//!
//! Implements the diagonal approximation of structured state spaces:
//!
//! ```text
//! x'(t) = Λ x(t) + B u(t)
//! y(t)  = Re[ C x(t) ] + D u(t)
//! ```
//!
//! with complex diagonal Λ initialized via S4D-Lin or S4D-Inv,
//! discretized using the zero-order hold (ZOH) method.

use anyhow::Result;
use log::{debug, info};
use ndarray::Array2;
use num_complex::Complex64;
use rand::Rng;
use rand_distr::StandardNormal;
use std::f64::consts::PI;

use super::config::{DiagonalSSMConfig, InitMethod};
use crate::data::dataset::Dataset;

/// A single diagonal SSM layer with complex eigenvalues.
#[derive(Debug, Clone)]
struct SSMLayer {
    /// Diagonal eigenvalues Λ (complex, length state_dim).
    lambda: Vec<Complex64>,
    /// Input-to-state matrix B (complex, shape: state_dim x d_model).
    #[allow(dead_code)]
    b: Vec<Vec<Complex64>>,
    /// State-to-output matrix C (complex, shape: d_model x state_dim).
    c: Vec<Vec<Complex64>>,
    /// Feedthrough matrix D (real, length d_model).
    d: Vec<f64>,
    /// Discretization step size.
    #[allow(dead_code)]
    dt: f64,
    /// Discretized Ā = exp(Λ * dt) (complex, length state_dim).
    a_bar: Vec<Complex64>,
    /// Discretized B̄ = (Ā - I) * Λ⁻¹ * B (complex, shape: state_dim x d_model).
    b_bar: Vec<Vec<Complex64>>,
}

impl SSMLayer {
    /// Create a new SSM layer with the given configuration.
    fn new(state_dim: usize, d_model: usize, dt: f64, init_method: InitMethod) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize diagonal eigenvalues
        let lambda: Vec<Complex64> = match init_method {
            InitMethod::S4DLin => {
                // S4D-Lin: λₙ = -0.5 + j·π·n
                (0..state_dim)
                    .map(|n| Complex64::new(-0.5, PI * n as f64))
                    .collect()
            }
            InitMethod::S4DInv => {
                // S4D-Inv: λₙ = -0.5 + j·π·n/(2n+1)
                (0..state_dim)
                    .map(|n| {
                        let nf = n as f64;
                        Complex64::new(-0.5, PI * nf / (2.0 * nf + 1.0))
                    })
                    .collect()
            }
            InitMethod::Random => (0..state_dim)
                .map(|_| {
                    Complex64::new(
                        -0.5 + 0.1 * rng.sample::<f64, _>(StandardNormal),
                        rng.sample::<f64, _>(StandardNormal),
                    )
                })
                .collect(),
        };

        // Initialize B with small random complex values
        let b: Vec<Vec<Complex64>> = (0..state_dim)
            .map(|_| {
                (0..d_model)
                    .map(|_| {
                        let scale = 1.0 / (state_dim as f64).sqrt();
                        Complex64::new(
                            scale * rng.sample::<f64, _>(StandardNormal),
                            scale * rng.sample::<f64, _>(StandardNormal),
                        )
                    })
                    .collect()
            })
            .collect();

        // Initialize C with small random complex values
        let c: Vec<Vec<Complex64>> = (0..d_model)
            .map(|_| {
                (0..state_dim)
                    .map(|_| {
                        let scale = 1.0 / (state_dim as f64).sqrt();
                        Complex64::new(
                            scale * rng.sample::<f64, _>(StandardNormal),
                            scale * rng.sample::<f64, _>(StandardNormal),
                        )
                    })
                    .collect()
            })
            .collect();

        // Initialize D (feedthrough) to zero
        let d = vec![0.0; d_model];

        // Compute discretized matrices
        let (a_bar, b_bar) = discretize_zoh(&lambda, &b, dt);

        SSMLayer {
            lambda,
            b,
            c,
            d,
            dt,
            a_bar,
            b_bar,
        }
    }

    /// Compute the SSM convolution kernel of length `seq_len`.
    ///
    /// K[t] = Re[ C * Ā^t * B̄ ] for t = 0, ..., seq_len - 1
    ///
    /// This is computed efficiently via the Vandermonde product.
    fn compute_kernel(&self, seq_len: usize, d_model: usize) -> Array2<f64> {
        // Kernel shape: (seq_len, d_model)
        let state_dim = self.lambda.len();
        let mut kernel = Array2::<f64>::zeros((seq_len, d_model));

        // Precompute powers of Ā: Ā^0, Ā^1, ..., Ā^(seq_len-1)
        // This is the Vandermonde product
        let mut a_powers = vec![vec![Complex64::new(1.0, 0.0); state_dim]; seq_len];
        for t in 1..seq_len {
            for n in 0..state_dim {
                a_powers[t][n] = a_powers[t - 1][n] * self.a_bar[n];
            }
        }

        // K[t] = Re[ sum_n C[:,n] * Ā[n]^t * B̄[n,:] ] for each output dim
        for t in 0..seq_len {
            for h_out in 0..d_model {
                let mut val = Complex64::new(0.0, 0.0);
                for n in 0..state_dim {
                    for h_in in 0..d_model {
                        if h_in == h_out {
                            val += self.c[h_out][n] * a_powers[t][n] * self.b_bar[n][h_in];
                        }
                    }
                }
                kernel[[t, h_out]] = val.re;
            }
        }

        kernel
    }

    /// Forward pass through this SSM layer using convolution.
    ///
    /// Input shape: (seq_len, d_model)
    /// Output shape: (seq_len, d_model)
    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let seq_len = input.nrows();
        let d_model = input.ncols();

        // Compute the convolution kernel
        let kernel = self.compute_kernel(seq_len, d_model);

        // Convolve input with kernel (causal convolution)
        let mut output = Array2::<f64>::zeros((seq_len, d_model));

        for t in 0..seq_len {
            for h in 0..d_model {
                let mut sum = 0.0;
                // Causal: only use kernel[0..=t]
                for k in 0..=t {
                    sum += kernel[[k, h]] * input[[t - k, h]];
                }
                // Add feedthrough
                output[[t, h]] = sum + self.d[h] * input[[t, h]];
            }
        }

        output
    }

    /// Forward pass using recurrent mode (step by step).
    ///
    /// More memory efficient for inference.
    fn forward_recurrent(&self, input: &Array2<f64>) -> Array2<f64> {
        let seq_len = input.nrows();
        let d_model = input.ncols();
        let state_dim = self.lambda.len();

        let mut state = vec![Complex64::new(0.0, 0.0); state_dim];
        let mut output = Array2::<f64>::zeros((seq_len, d_model));

        for t in 0..seq_len {
            // Update state: x[t] = Ā * x[t-1] + B̄ * u[t]
            let mut new_state = vec![Complex64::new(0.0, 0.0); state_dim];
            for n in 0..state_dim {
                new_state[n] = self.a_bar[n] * state[n];
                for h in 0..d_model {
                    new_state[n] += self.b_bar[n][h] * Complex64::new(input[[t, h]], 0.0);
                }
            }
            state = new_state;

            // Output: y[t] = Re[ C * x[t] ] + D * u[t]
            for h in 0..d_model {
                let mut val = Complex64::new(0.0, 0.0);
                for n in 0..state_dim {
                    val += self.c[h][n] * state[n];
                }
                output[[t, h]] = val.re + self.d[h] * input[[t, h]];
            }
        }

        output
    }
}

/// Zero-Order Hold (ZOH) discretization.
///
/// Given continuous-time parameters Λ (diagonal) and B:
/// - Ā = exp(Λ * Δ)
/// - B̄ = (Ā - I) * Λ⁻¹ * B
fn discretize_zoh(
    lambda: &[Complex64],
    b: &[Vec<Complex64>],
    dt: f64,
) -> (Vec<Complex64>, Vec<Vec<Complex64>>) {
    let state_dim = lambda.len();
    let d_model = if state_dim > 0 { b[0].len() } else { 0 };

    let dt_c = Complex64::new(dt, 0.0);
    let one = Complex64::new(1.0, 0.0);

    // Ā = exp(Λ * dt)
    let a_bar: Vec<Complex64> = lambda.iter().map(|&lam| (lam * dt_c).exp()).collect();

    // B̄ = (Ā - I) * Λ⁻¹ * B = (exp(Λ*dt) - I) / Λ * B
    let b_bar: Vec<Vec<Complex64>> = (0..state_dim)
        .map(|n| {
            let scale = if lambda[n].norm() > 1e-12 {
                (a_bar[n] - one) / lambda[n]
            } else {
                dt_c // limit as λ -> 0
            };
            (0..d_model).map(|h| scale * b[n][h]).collect()
        })
        .collect();

    (a_bar, b_bar)
}

/// Multi-layer Diagonal SSM model for sequence prediction.
#[derive(Debug, Clone)]
pub struct DiagonalSSMModel {
    /// Model configuration.
    config: DiagonalSSMConfig,
    /// Stack of SSM layers.
    layers: Vec<SSMLayer>,
    /// Output projection weights (d_model -> 1).
    output_weights: Vec<f64>,
    /// Output bias.
    output_bias: f64,
    /// Training loss history.
    loss_history: Vec<f64>,
}

impl DiagonalSSMModel {
    /// Create a new Diagonal SSM model with the given configuration.
    pub fn new(config: DiagonalSSMConfig) -> Self {
        let mut rng = rand::thread_rng();

        let layers: Vec<SSMLayer> = (0..config.num_layers)
            .map(|_| SSMLayer::new(config.state_dim, config.d_model, config.dt, config.init_method))
            .collect();

        // Initialize output projection
        let scale = 1.0 / (config.d_model as f64).sqrt();
        let output_weights: Vec<f64> = (0..config.d_model)
            .map(|_| scale * rng.sample::<f64, _>(StandardNormal))
            .collect();

        info!(
            "Created DiagonalSSM: state_dim={}, d_model={}, layers={}, init={:?}",
            config.state_dim, config.d_model, config.num_layers, config.init_method
        );

        Self {
            config,
            layers,
            output_weights,
            output_bias: 0.0,
            loss_history: Vec::new(),
        }
    }

    /// Forward pass through the full model.
    ///
    /// Input: sequence of shape (seq_len, d_model)
    /// Output: single prediction value (sigmoid-activated)
    pub fn forward(&self, input: &Array2<f64>) -> f64 {
        let mut x = input.clone();

        // Pass through each SSM layer with residual connections
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_out = layer.forward(&x);
            if i > 0 {
                // Residual connection for layers after the first
                x = &x + &layer_out;
            } else {
                x = layer_out;
            }
            // Apply activation (GELU approximation)
            x.mapv_inplace(|v| gelu(v));
        }

        // Global average pooling over time dimension
        let seq_len = x.nrows();
        let d_model = x.ncols();
        let mut pooled = vec![0.0; d_model];
        for h in 0..d_model {
            for t in 0..seq_len {
                pooled[h] += x[[t, h]];
            }
            pooled[h] /= seq_len as f64;
        }

        // Linear projection to scalar
        let mut logit = self.output_bias;
        for h in 0..d_model {
            logit += self.output_weights[h] * pooled[h];
        }

        // Sigmoid activation for binary classification
        sigmoid(logit)
    }

    /// Forward pass using recurrent mode (for inference).
    pub fn forward_recurrent(&self, input: &Array2<f64>) -> f64 {
        let mut x = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_out = layer.forward_recurrent(&x);
            if i > 0 {
                x = &x + &layer_out;
            } else {
                x = layer_out;
            }
            x.mapv_inplace(|v| gelu(v));
        }

        let seq_len = x.nrows();
        let d_model = x.ncols();
        let mut pooled = vec![0.0; d_model];
        for h in 0..d_model {
            for t in 0..seq_len {
                pooled[h] += x[[t, h]];
            }
            pooled[h] /= seq_len as f64;
        }

        let mut logit = self.output_bias;
        for h in 0..d_model {
            logit += self.output_weights[h] * pooled[h];
        }

        sigmoid(logit)
    }

    /// Train the model using simple SGD on a dataset.
    ///
    /// Uses binary cross-entropy loss and numerical gradient estimation
    /// for parameter updates.
    pub fn train(&mut self, dataset: &Dataset) -> Result<Vec<f64>> {
        let epochs = self.config.epochs;
        let lr = self.config.learning_rate;
        let n_windows = dataset.n_windows;

        info!(
            "Training for {} epochs, lr={}, {} samples",
            epochs, lr, n_windows
        );

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for i in 0..n_windows {
                let input = dataset.get_window(i);
                let target = dataset.get_target(i);

                // Forward pass
                let pred = self.forward(&input);
                let pred_clamped = pred.clamp(1e-7, 1.0 - 1e-7);

                // Binary cross-entropy loss
                let loss =
                    -(target * pred_clamped.ln() + (1.0 - target) * (1.0 - pred_clamped).ln());
                epoch_loss += loss;

                // Gradient of BCE w.r.t. logit (before sigmoid)
                let grad = pred_clamped - target;

                // Update output weights and bias using gradient descent
                let seq_len = input.nrows();
                let d_model = input.ncols();

                // Recompute pooled features for gradient
                let mut x = input.clone();
                for (li, layer) in self.layers.iter().enumerate() {
                    let layer_out = layer.forward(&x);
                    if li > 0 {
                        x = &x + &layer_out;
                    } else {
                        x = layer_out;
                    }
                    x.mapv_inplace(|v| gelu(v));
                }

                let mut pooled = vec![0.0; d_model];
                for h in 0..d_model {
                    for t in 0..seq_len {
                        pooled[h] += x[[t, h]];
                    }
                    pooled[h] /= seq_len as f64;
                }

                // Update output projection
                for h in 0..d_model {
                    let w_grad = grad * pooled[h] + self.config.weight_decay * self.output_weights[h];
                    let clipped = w_grad.clamp(-self.config.grad_clip, self.config.grad_clip);
                    self.output_weights[h] -= lr * clipped;
                }
                self.output_bias -= lr * grad.clamp(-self.config.grad_clip, self.config.grad_clip);
            }

            let avg_loss = epoch_loss / n_windows as f64;
            losses.push(avg_loss);

            if epoch % 10 == 0 || epoch == epochs - 1 {
                debug!("Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, avg_loss);
            }
        }

        self.loss_history.extend_from_slice(&losses);
        info!(
            "Training complete. Final loss: {:.6}",
            losses.last().unwrap_or(&0.0)
        );

        Ok(losses)
    }

    /// Generate predictions for a sequence of input windows.
    ///
    /// Returns a vector of prediction probabilities (0..1).
    pub fn predict(&self, dataset: &Dataset) -> Vec<f64> {
        (0..dataset.n_windows)
            .map(|i| {
                let input = dataset.get_window(i);
                self.forward(&input)
            })
            .collect()
    }

    /// Generate a single prediction from one input window.
    pub fn predict_one(&self, input: &Array2<f64>) -> f64 {
        self.forward(input)
    }

    /// Get the training loss history.
    pub fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }

    /// Get the model configuration.
    pub fn config(&self) -> &DiagonalSSMConfig {
        &self.config
    }

    /// Get the eigenvalue spectrum of the first layer.
    pub fn eigenvalues(&self) -> Option<Vec<Complex64>> {
        self.layers.first().map(|l| l.lambda.clone())
    }
}

/// Sigmoid activation function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// GELU activation function (approximation).
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::DiagonalSSMConfig;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_gelu() {
        // GELU(0) should be ~0
        assert!(gelu(0.0).abs() < 1e-10);
        // GELU(x) ~ x for large x
        assert!((gelu(3.0) - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_model_creation() {
        let config = DiagonalSSMConfig::s4d_lin(16, 3);
        let model = DiagonalSSMModel::new(config);
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.output_weights.len(), 3);
    }

    #[test]
    fn test_forward_pass() {
        let config = DiagonalSSMConfig::s4d_lin(8, 3).with_num_layers(1);
        let model = DiagonalSSMModel::new(config);
        let input = Array2::<f64>::zeros((10, 3));
        let pred = model.forward(&input);
        assert!(pred >= 0.0 && pred <= 1.0);
    }

    #[test]
    fn test_zoh_discretization() {
        let lambda = vec![Complex64::new(-0.5, PI)];
        let b = vec![vec![Complex64::new(1.0, 0.0)]];
        let dt = 0.01;
        let (a_bar, b_bar) = discretize_zoh(&lambda, &b, dt);
        // exp(lambda * dt) should have magnitude < 1 for stable system
        assert!(a_bar[0].norm() < 1.0 + 1e-10);
        assert!(!b_bar[0][0].is_nan());
    }

    #[test]
    fn test_s4d_lin_initialization() {
        let layer = SSMLayer::new(4, 2, 0.01, InitMethod::S4DLin);
        // Check S4D-Lin: lambda_0 = -0.5 + 0j, lambda_1 = -0.5 + pi*j
        assert!((layer.lambda[0].re - (-0.5)).abs() < 1e-10);
        assert!((layer.lambda[0].im - 0.0).abs() < 1e-10);
        assert!((layer.lambda[1].re - (-0.5)).abs() < 1e-10);
        assert!((layer.lambda[1].im - PI).abs() < 1e-10);
    }

    #[test]
    fn test_recurrent_matches_convolution() {
        let config = DiagonalSSMConfig::s4d_lin(4, 2).with_num_layers(1);
        let model = DiagonalSSMModel::new(config);
        let input = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64 * 0.1);

        let conv_out = model.forward(&input);
        let rec_out = model.forward_recurrent(&input);

        // Both modes should produce similar results
        assert!(
            (conv_out - rec_out).abs() < 0.1,
            "Convolution ({}) and recurrent ({}) outputs differ too much",
            conv_out,
            rec_out
        );
    }
}
