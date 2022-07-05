use ndarray::*;

extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

pub mod fact;

pub struct NN {
    _topology : Vec<usize>,
    _activation_functions : Vec<String>,
    _learning_rate : f64,
    _model : Vector<NNLayer>,
    _debug : bool
}

impl NN {
    // Construct NN
    pub fn new(topology : Vec<usize>, activation_functions : Vec<String>, learning_rate : f64, debug : bool) -> NN{
        let mut nn = NN {
            _topology : topology,
            _activation_functions : activation_functions.clone(),
            _learning_rate : learning_rate,
            _model : vec![]
            _debug : debug
        };

        for i in 0..topology.len() - 1 {
            nn._model.push(nnlayer::new(topology[i]+1, topology[i+1],act_funcs[i], learning_rate));            
        }

        return nn;
    }



    // Forward propagation
    pub fn forward(&mut self, input : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        let mut vals = na::DMatrix::from_element(input_size + 1, 1, 0.);
        vals = input.clone();

        for x in self._model {
            vals = x.forward(vals.clone());
        }

        return vals;
    }



    // Back propagation
    pub fn backward(&mut self, actions : na::DMatrix::<f64>, experimentals : na::DMatrix::<f64>) {
        let mut delta = na::DMatrix::from_element(actions.nrows(), 1, 0.);
        delta = actions.clone() - experimentals.clone();

        for i in 0..self.._model.len() - 1 {
            delta = self._model[i].backward(delta.clone());
        }
    }



    // Update time
    pub fn update_time(&mut self) {
        for x in self._model {
            x.update_time();
        }
    }



    // Train
    pub fn train(&mut self, num_episodes : i64, max_steps : i64, target_upd : i64, exp_upd : i64) {
        
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