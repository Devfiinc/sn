
use postgres::{Client, Error, NoTls};
use ndarray::*;

extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

use crate::fact;
use crate::dp;



pub struct LogisticRegression {
    _max_iter : usize,
    _nfeat : usize,
    _nclass : usize,
    _eta : f64,
    _mu : f64,

    _w : na::DMatrix::<f64>,

    // Differential privacy
    _dp : bool,
    _noise_scale : f64,
    _gradient_norm_bound : f64,

    // Debug
    _debug : bool


}



impl LogisticRegression {
    pub fn new(max_iter : usize, nfeat : usize, nclass : usize, eta : f64, mu : f64, debug : bool) -> LogisticRegression {
        let mut lr = LogisticRegression {

            _max_iter : max_iter,
            _nfeat : nfeat,
            _nclass : nclass,
            _eta : eta,
            _mu : mu,

            //_w : na::DMatrix::from_fn(nfeat, nclass, |r,c| {rand::random::<f64>() - 0.5}),
            _w : na::DMatrix::from_element(nfeat, nclass, 0.),

            _dp : false,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,

            _debug : debug,


        };

        return lr;
    }


    pub fn enable_dp(&mut self, dp : bool, noise_scale : f64, gradient_norm_bound : f64) {
        self._dp = dp;
        self._noise_scale = noise_scale;
        self._gradient_norm_bound = gradient_norm_bound;
    }


    pub fn disable_dp(&mut self) {
        self._dp = false;
    }


    pub fn loss(&mut self, x : na::DMatrix::<f64>, y : na::DMatrix::<f64>) -> f64 {
        let z = - x.clone() * self._w.clone();
        //println!("{} {}", x.nrows(), x.ncols());

        let m = x.nrows() as f64;

        let mut expsum : f64 = 0.0;
        for i in 0..z.nrows() {
            for j in 0..z.ncols() {
                expsum += z[(i,j)].exp();
            }
        }
        expsum = expsum.clone();
        
        let lossmat = (1.0 / m) * ( ( x * self._w.clone() ) * y.transpose() );
        let loss = lossmat[(0,0)] as f64 + expsum.ln();

        return loss;
    }


    fn gradient(&mut self, x : na::DMatrix::<f64>, y : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        self.print_dmatrix(x.clone(), "x");
        self.print_dmatrix(y.clone(), "y");
        let z = - x.clone() * self._w.clone();
        self.print_dmatrix(z.clone(), "z");
        let a = fact::softmax(z.clone());
        self.print_dmatrix(a.clone(), "a");
        let m = x.nrows() as f64;

        let grad = (1.0 / m) * (x.transpose() * (y - a)) + 2.0 * self._mu * self._w.clone();
        self.print_dmatrix(grad.clone(), "grad");

        return grad;
    } 


    pub fn print_dmatrix (&mut self, x : na::DMatrix::<f64>, name : &str) {
        print!("{} = ", name);
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                print!("{} ", x[(i,j)]);
            }
            println!("");
        }
    }


    pub fn fit(&mut self, x_train : Vec<Vec<Option<f64>>>, y_train : Vec<Option<f64>>) {

        if x_train.len() < self._max_iter {
            self._max_iter = x_train.len();
        }


        //if self._w.nrows() != x_train[0].len() {
        //    self._w = na::DMatrix::from_fn(x_train[0].len(), 10, |r,c| {rand::random::<f64>() - 0.5});
        //}
        

        //println!("fit");

        for i in 0..1000 {
            for i in 0..self._max_iter {

                if (i % 1000) == 0 {
                    println!("Training = {:.2} %", 100.0 * i as f64 / x_train.len() as f64);
                }

                let mut li = na::DMatrix::<f64>::from_vec(1, x_train[0].len(), x_train[i].iter().map(|x| x.unwrap()).collect());
                
                let mut lo = na::DMatrix::from_element(1, self._nclass, 0.);
                lo[(0, y_train[i].unwrap() as usize)] = 1.0;

                let gradient = self.gradient(li.clone(), lo.clone());
                self._w = self._w.clone() - self._eta * gradient;
                let loss = self.loss(li.clone(), lo.clone());
                //println!("Loss = {:.2}", loss);
            }
        }
    }


    pub fn test(&mut self, x_test : Vec<Vec<Option<f64>>>, y_test : Vec<Option<f64>>) {

        let mut correct : i64 = 0;

        for i in 0..x_test.len() {
            if (i % 100) == 0 {
                println!("Testing = {:.2} %", 100.0 * i as f64 / x_test.len() as f64);
            }
    
            let li = na::DMatrix::<f64>::from_vec(1, x_test[0].len(), x_test[i].iter().map(|x| x.unwrap()).collect());

            let z = - li * self._w.clone();
            //let a = z.map(|x| fact::softmax(x));
            let a = fact::softmax(z.clone());
            let maxid = self.get_max_idx(a.clone());


            if maxid == y_test[i].unwrap() as i64 {
                correct = correct + 1;
            }
        }
    
        println!("Accuracy: {} / {} = {}%\n", correct, x_test.len(), (correct as f64 / x_test.len() as f64) * 100.);
    }


    

    pub fn get_max_idx(&self, mat : na::DMatrix::<f64>) -> i64 {
        let mut maxid : usize = 0;
        let mut maxval : f64 = 0.0;

        for i in 0..mat.nrows(){
            for j in 0..mat.ncols() {
                if mat[(i,j)] > maxval {
                    maxid = j;
                    maxval = mat[(i,j)];
                }
            }
        }

        return maxid as i64;
    }



    pub fn predict(&mut self, input : Vec<Option<f64>>) -> i64 {

        let li = na::DMatrix::<f64>::from_vec(1, input.len(), input.iter().map(|x| x.unwrap()).collect());
        
        let z = - li * self._w.clone();

        //let a = z.map(|x| fact::softmax(x));
        let a = fact::softmax(z.clone());

        let maxid = self.get_max_idx(a.clone());

        return maxid as i64;
    }


}

