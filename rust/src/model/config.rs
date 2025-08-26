//! Configuration for the Diagonal SSM model.

use serde::{Deserialize, Serialize};

/// Initialization method for the diagonal eigenvalues.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum InitMethod {
    /// S4D-Lin initialization: lambda_n = -0.5 + j * pi * n
    S4DLin,
    /// S4D-Inv initialization: lambda_n = -0.5 + j * pi * n / (2n + 1)
    S4DInv,
    /// Random initialization from normal distribution.
    Random,
}

impl Default for InitMethod {
    fn default() -> Self {
        Self::S4DLin
    }
}

/// Configuration for the Diagonal SSM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagonalSSMConfig {
    /// Dimension of the state space (N).
    pub state_dim: usize,
    /// Input/output model dimension (H).
    pub d_model: usize,
    /// Number of SSM layers stacked.
    pub num_layers: usize,
    /// Learning rate for SGD.
    pub learning_rate: f64,
    /// Discretization step size.
    pub dt: f64,
    /// Initialization method for eigenvalues.
    pub init_method: InitMethod,
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size for training.
    pub batch_size: usize,
    /// L2 regularization coefficient.
    pub weight_decay: f64,
    /// Gradient clipping threshold.
    pub grad_clip: f64,
}

impl Default for DiagonalSSMConfig {
    fn default() -> Self {
        Self {
            state_dim: 64,
            d_model: 5,
            num_layers: 2,
            learning_rate: 0.001,
            dt: 0.01,
            init_method: InitMethod::S4DLin,
            epochs: 100,
            batch_size: 32,
            weight_decay: 1e-4,
            grad_clip: 1.0,
        }
    }
}

impl DiagonalSSMConfig {
    /// Create a new config with S4D-Lin initialization.
    pub fn s4d_lin(state_dim: usize, d_model: usize) -> Self {
        Self {
            state_dim,
            d_model,
            init_method: InitMethod::S4DLin,
            ..Default::default()
        }
    }

    /// Create a new config with S4D-Inv initialization.
    pub fn s4d_inv(state_dim: usize, d_model: usize) -> Self {
        Self {
            state_dim,
            d_model,
            init_method: InitMethod::S4DInv,
            ..Default::default()
        }
    }

    /// Set the learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the number of epochs.
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set the number of layers.
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set the discretization step.
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }
}
