use ndarray::*;

extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

pub struct NN {
    _topology : Vec<usize>,
    _learning_rate : f64,
    _debug : bool
}

impl NN {
    // Construct NN
    pub fn new(topology : Vec<usize>, learning_rate : f64, debug : bool) -> NN{
        NN {
            _topology : topology,
            _learning_rate : learning_rate,
            _debug : debug
        }
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

    pub fn tanh(x : f64) -> f64 {
        x.tanh()
    }

    pub fn relu(x : f64) -> f64 {
        if x < 0. {
            0.
        } else {
            x
        }
    }

    pub fn softmax(x : f64) -> f64 {
        x.exp()
    }

    pub fn softplus(x : f64) -> f64 {
        x.ln()
    }

    pub fn softsign(x : f64) -> f64 {
        x / (1. + x.abs())
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

    pub fn exponential(x : f64) -> f64 {
        x.exp()
    }

    pub fn linear(x : f64) -> f64 {
        x
    }

    pub fn leaky_relu(x : f64) -> f64 {
        if x < 0. {
            0.01 * x
        } else {
            x
        }
    }

    pub fn elu(x : f64) -> f64 {
        if x < 0. {
            0.01 * (x.exp() - 1.)
        } else {
            x
        }
    }

    pub fn selu(x : f64) -> f64 {
        if x < 0. {
            1.6732632423543772848170429916717 * (x.exp() - 1.)
        } else {
            x
        }
    }

}