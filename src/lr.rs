
use postgres::{Client, Error, NoTls};
use ndarray::*;

extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

use crate::fact;
use crate::dp;



pub struct LogisticRegression {
    _max_iter : i64,
    _eta : f64,
    _mu : f64,

    _w : na::DMatrix::<f64>,

    // Differential privacy
    _dp : bool,
    _noise_scale : f64,
    _gradient_norm_bound : f64,

    // Debug
    _debug : bool,


}



impl LogisticRegression {
    pub fn new(max_iter : i64, debug : bool) -> LogisticRegression {
        let mut lr = LogisticRegression {

            _max_iter : max_iter,

            _dp : false,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,

            _debug : debug,


        };

        return lr;
    };


    pub fn enable_dp(&mut self, dp : bool, noise_scale : f64, gradient_norm_bound : f64) {
        self._dp = dp;
        self._noise_scale = noise_scale;
        self._gradient_norm_bound = gradient_norm_bound;
    }


    pub fn disable_dp(&mut self) {
        self._dp = false;
    }

    //fn loss(&mut self, x : na::DMatrix::<f64>, y : na::DMatrix::<f64>, w : na::DMatrix::<f64>) -> f64 {
    //    let z = - x * w;
    //    let m = x.nrows(); 
    //}

    fn gradient(&mut self, x : na::DMatrix::<f64>, y : na::DMatrix::<f64>, w : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        let z = - x.transpose() * w;

        let a = z.map(|x| fact::softmax(x));

        let m = x.nrows();

        let grad = (1.0 / m) * (x.transpose() * (y - a)) * + 2 self._mu * w;

        return grad;
    } 

    pub fn fit(&mut self, x_train : Vec<Vec<Option<f64>>>, y_train : Vec<Option<f64>>) {

        if x_train.len() < self._max_iter {
            self._max_iter = x_train.len();
        }

        let mut w = na::DMatrix::from_element(x_train[0].len(), 10, 0.);

        for i in 0..self._max_iter {

            if (i % 1000) == 0 {
                println!("Training = {:.2} %", 100.0 * i as f64 / x_train.len() as f64);
            }

            let mu li = na::DMatrix::<f64>::from_vec(self._topology[0], 1, x_train[i].iter().map(|x| x.unwrap()).collect());
            
            let mut lo = na::DMatrix::from_element(10, 1, 0.);
            lo[(y_test[i].unwrap() as usize, 0)] = 1.0;

            let gradient = gradient(li.clone(), lo.clone(), w.clone());
            w = w - self._eta * gradient;
            //let loss = loss(x.clone(), y.clone(), w.clone());
        }

    }

    pub fn predict(&mut self, input : Vec<Option<f64>>) -> {

        let li = na::DMatrix::<f64>::from_vec(input.len(), 1, input.iter().map(|x| x.unwrap()).collect());
        
        let z = - li * w;

        let a = z.map(|x| fact::softmax(x));

        
        let mut maxid : usize = 0;
        let mut maxval : f64 = 0.0;
        for j in 0..lo.ncols(){
            if lo[(0,j)] > maxval {
                maxid = j;
                maxval = lo[(0,j)];
            }
        }
        return maxid as i64;

    }


}

