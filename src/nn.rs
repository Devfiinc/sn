use ndarray::*;

extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

pub struct NN {
    _topology : Vec<usize>,
    _learning_rate : f64,
    _debug : bool,
    _neuron_layers : Vec<na::DMatrix::<f64>>,
    _cache_layers : Vec<na::DMatrix::<f64>>,
    _deltas : Vec<na::DMatrix::<f64>>,
    _weights : Vec<na::DMatrix::<f64>>
}

impl NN {
    // Construct NN
    pub fn new(topology : Vec<usize>, learning_rate : f64, debug : bool) -> NN{
        let mut nn = NN {
            _topology : topology,
            _learning_rate : learning_rate,
            _debug : debug,
            _neuron_layers : vec![],
            _cache_layers : vec![],
            _deltas : vec![],
            _weights : vec![]
        };

        for i in 0..nn._topology.len() {
            if i == nn._topology.len() -1 {
                nn._neuron_layers.push(na::DMatrix::from_element(nn._topology[i], 1, 0.))
            }
            else {
                nn._neuron_layers.push(DMatrix::from_element(nn._topology[i], 1, 0.));
                nn._cache_layers.push(DMatrix::from_element(nn._topology[i], 1, 0.));
                nn._deltas.push(DMatrix::from_element(nn._topology[i], 1, 0.));
            }
        }

        return nn;
    }

    pub fn get_learning_rate(&self) -> f64{
        return self._learning_rate;
    }


    /*

    // Forward propagation
    pub fn propagate_forward() {

    }

    // Back propagation
    pub fn propagate_backward() {

    }

    // Calculate errors
    pub fn calculate_errors() {

    }

    // Update weights
    pub fn update_weights() {

    }

    // Train
    pub fn train() {

    }

    // Predict
    pub fn predict() {

    }

    */





    // Activation functions
    pub fn sigmoid(x : f64) -> f64 {
        1. / (1. + (-x).exp())
    }

    pub fn sigmoid_derivative(x : f64) -> f64 {
        let sigmoid_x = NN::sigmoid(x);
        sigmoid_x * (1. - sigmoid_x)
    }

    pub fn tanh(x : f64) -> f64 {
        x.tanh()
    }

    pub fn tanh_derivative(x : f64) -> f64 {
        let tanh_x = NN::tanh(x);
        1. - tanh_x * tanh_x
    }

    pub fn relu(x : f64) -> f64 {
        if x < 0. {
            0.
        } else {
            x
        }
    }

    pub fn relu_derivative(x : f64) -> f64 {
        if x < 0. {
            0.
        } else {
            1.
        }
    }

    pub fn softmax(x : f64) -> f64 {
        x.exp()
    }

    pub fn softmax_derivative(x : f64) -> f64 {
        let softmax_x = NN::softmax(x);
        softmax_x * (1. - softmax_x)
    }

    pub fn softplus(x : f64) -> f64 {
        x.ln()
    }

    pub fn softplus_derivative(x : f64) -> f64 {
        let softplus_x = NN::softplus(x);
        softplus_x * (1. - softplus_x)
    }

    pub fn softsign(x : f64) -> f64 {
        x / (1. + x.abs())
    }

    pub fn softsign_derivative(x : f64) -> f64 {
        let softsign_x = NN::softsign(x);
        softsign_x * (1. - softsign_x)
    }

    pub fn hard_sigmoid(x : f64) -> f64 {
        if x < 0. {
            0.
        } else if x > 1. {
            1.
        } else {
            x
        }
    }

    pub fn hard_sigmoid_derivative(x : f64) -> f64 {
        if x < 0. {
            0.
        } else if x > 1. {
            0.
        } else {
            1.
        }
    }

    pub fn exponential(x : f64) -> f64 {
        x.exp()
    }

    pub fn exponential_derivative(x : f64) -> f64 {
        let exponential_x = NN::exponential(x);
        exponential_x * (1. - exponential_x)
    }

    pub fn linear(x : f64) -> f64 {
        x
    }

    pub fn linear_derivative(x : f64) -> f64 {
        1.
    }

    pub fn leaky_relu(x : f64) -> f64 {
        if x < 0. {
            0.01 * x
        } else {
            x
        }
    }

    pub fn leaky_relu_derivative(x : f64) -> f64 {
        if x < 0. {
            0.01
        } else {
            1.
        }
    }

    pub fn elu(x : f64) -> f64 {
        if x < 0. {
            0.01 * (x.exp() - 1.)
        } else {
            x
        }
    }

    pub fn elu_derivative(x : f64) -> f64 {
        if x < 0. {
            0.01 * x.exp()
        } else {
            1.
        }
    }

    pub fn selu(x : f64) -> f64 {
        if x < 0. {
            1.6732632423543772848170429916717 * (x.exp() - 1.)
        } else {
            x
        }
    }

    pub fn selu_derivative(x : f64) -> f64 {
        if x < 0. {
            1.6732632423543772848170429916717 * x.exp()
        } else {
            1.
        }
    }

}