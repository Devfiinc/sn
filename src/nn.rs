use crate::nnlayer;


use ndarray::*;
extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

pub struct NN {
    _topology : Vec<usize>,
    _activation_functions : Vec<String>,
    _learning_rate : f64,
    _model : Vec<nnlayer::NNLayer>,
    _debug : bool
}

impl NN {
    // Construct NN
    pub fn new(topology : Vec<usize>, activation_functions : Vec<String>, learning_rate : f64, debug : bool) -> NN{
        let mut nn = NN {
            _topology : topology.clone(),
            _activation_functions : activation_functions.clone(),
            _learning_rate : learning_rate,
            _model : vec![],
            _debug : debug
        };

        for i in 0..topology.len() - 1 {
            let fact = activation_functions[i].clone();
            nn._model.push(nnlayer::NNLayer::new(topology[i] + 1, topology[i+1], fact, learning_rate, debug));            
        }

        return nn;
    }



    // Forward propagation
    pub fn forward(&mut self, input : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        let mut vals = na::DMatrix::from_element(input.len() + 1, 1, 0.);
        vals = input.clone();

        for i in 0..self._model.len() {
            vals = self._model[i].forward(vals.clone());
        }

        return vals;
    }



    // Back propagation
    pub fn backward(&mut self, actions : na::DMatrix::<f64>, experimentals : na::DMatrix::<f64>) {
        let mut delta = na::DMatrix::from_element(actions.nrows(), 1, 0.);
        delta = actions.clone() - experimentals.clone();

        for i in 0..self._model.len() - 1 {
            delta = self._model[i].backward(delta.clone());
        }
    }



    // Update time
    pub fn update_time(&mut self) {
        for i in 0..self._model.len() {
            self._model[i].update_time();
        }
    }



    // Train
    pub fn train(&mut self, num_episodes : i64, max_steps : i64, target_upd : i64, exp_upd : i64) {
        let mut observation = na::DMatrix::from_element(2, 1, 0.);
        let mut observation1 = na::DMatrix::from_element(2, 1, 0.);
        
        let mut action : i64;
        let mut reward : f64 = 0.0;

        for episode in 0..num_episodes {
            
        }





    }



    // Test
    // pub fn test(max_steps : i64, num_episodes : i64, verbose : bool) {
    // 
    // }

    

    // Predict
    // pub fn predict() {
    // 
    // }



    pub fn get_learning_rate(&self) -> f64{
        return self._learning_rate;
    }


    pub fn debug_mode(&mut self, debug : bool) {
        self._debug = debug;
    }

}