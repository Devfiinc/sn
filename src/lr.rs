
use postgres::{Client, Error, NoTls};
use ndarray::*;

extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

use crate::fact;
use crate::dp;



pub struct LogisticRegression {
    _max_iter : usize,
    _batch_size : usize,
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
    pub fn new(max_iter : usize, batch_size : usize, nfeat : usize, nclass : usize, eta : f64, mu : f64, debug : bool) -> LogisticRegression {
        let mut lr = LogisticRegression {

            _max_iter : max_iter,
            _batch_size : batch_size,
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



    pub fn trace(&self, x : na::DMatrix::<f64>) -> f64 {
        let mut trace = 0.0;
        for i in 0..x.nrows() {
            trace = trace + x[(i,i)];
        }
        return trace;
    }

    pub fn loss(&mut self, x : na::DMatrix::<f64>, y : na::DMatrix::<f64>) -> f64 {
        let z = - x.clone() * self._w.clone();
        let m = x.nrows() as f64;

        let lossmat = self.trace(((x.clone() * self._w.clone() * y.transpose())).clone());

        let mut expsum1 = na::DMatrix::from_element(z.nrows(), 1, 0.);
        for i in 0..z.nrows() {
            let mut sumi : f64 = 0.0;
            for j in 0..z.ncols() {
                sumi = sumi + z[(i,j)].exp();
            }
            expsum1[(i,0)] = sumi;
        }

        let mut expsum : f64 = 0.0;
        for i in 0..expsum1.nrows() {
            expsum = expsum + expsum1[(i,0)].ln();
        }

        let loss = (1.0 / m) * (lossmat + expsum);

        return loss;
    }


    fn gradient(&mut self, x : na::DMatrix::<f64>, y : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        let z = - x.clone() * self._w.clone();
        let a = fact::softmax_mul(z.clone());
        let m = x.nrows() as f64;

        let grad = (1.0 / m) * (x.transpose() * (y - a)) + 2.0 * self._mu * self._w.clone();

        return grad;
    } 


    pub fn print_dmatrix (&mut self, x : na::DMatrix::<f64>, name : &str) {
        print!("{:.8} = ", name);
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                print!("{} ", x[(i,j)]);
            }
            println!("");
        }
    }


    pub fn fit(&mut self, x_train : Vec<Vec<Option<f64>>>, y_train : Vec<Option<f64>>) {
        
        let mut li = na::DMatrix::from_element(x_train.len(), x_train[0].len(), 0.);
        //println!("fit");
        for i in 0..x_train.len() {
            let mut lii = na::DMatrix::<f64>::from_vec(1, x_train[0].len(), x_train[i].iter().map(|x| x.unwrap()).collect());
            for j in 0..lii.ncols() {
                li[(i,j)] = lii[(0,j)];
            }
        }

        let mut lo = na::DMatrix::from_element(x_train.len(), self._nclass, 0.);
        for i in 0..y_train.len() {
            let mut loi = na::DMatrix::from_element(1, self._nclass, 0.);
            loi[(0, y_train[i].unwrap() as usize)] = 1.0;
            for j in 0..loi.ncols() {
                lo[(i,j)] = loi[(0,j)];
            }
        }

        let mut batch_idx : usize = 0;
        let mut batch_size : usize = self._batch_size;


        for i in 0..self._max_iter {
            batch_idx = 0;
            batch_size = self._batch_size;

            while batch_idx + batch_size < x_train.len() {

                let mut x_batch = na::DMatrix::from_element(batch_size, x_train[0].len(), 0.);
                let mut y_batch = na::DMatrix::from_element(batch_size, self._nclass, 0.);

                for j in 0..batch_size {
                    for k in 0..x_train[0].len() {
                        x_batch[(j,k)] = li[(batch_idx + j,k)];
                    }
                    for k in 0..self._nclass {
                        y_batch[(j,k)] = lo[(batch_idx + j,k)];
                    }
                }

                batch_idx = batch_idx + batch_size;

                let grad = self.gradient(x_batch.clone(), y_batch.clone());

                self._w = self._w.clone() - self._eta * grad;

                let loss = self.loss(x_batch.clone(),y_batch.clone());

                println!("Loss {:?} = {:.8}", i, loss);
            }


            if x_train.len() - batch_idx > 0 {
                batch_size = x_train.len() - batch_idx;
                //println!("Batch size = {}", batch_size);

                let mut x_batch = na::DMatrix::from_element(batch_size, x_train[0].len(), 0.);
                let mut y_batch = na::DMatrix::from_element(batch_size, self._nclass, 0.);

                for j in 0..batch_size {
                    for k in 0..x_train[0].len() {
                        x_batch[(j,k)] = li[(batch_idx + j,k)];
                    }
                    for k in 0..self._nclass {
                        y_batch[(j,k)] = lo[(batch_idx + j,k)];
                    }
                }

                batch_idx = batch_idx + batch_size;


                let grad = self.gradient(x_batch.clone(), y_batch.clone());

                self._w = self._w.clone() - self._eta * grad;

                let loss = self.loss(x_batch.clone(),y_batch.clone());

                println!("Loss {:?} = {:.8}", i, loss);
            }


            //let gradient = self.gradient(li.clone(), lo.clone());
            //self._w = self._w.clone() - self._eta * gradient;
            //let loss = self.loss(li.clone(), lo.clone());
            //println!("Loss {:?} = {:.8}", i, loss);
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

