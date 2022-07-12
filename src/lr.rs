extern crate nalgebra as na;

use crate::fact;
use crate::dp;





pub struct LogisticRegression {
    _max_iter : usize,
    _batch_size : usize,
    _nfeat : usize,
    _nclass : usize,
    _eta : f64,
    _mu : f64,
    _loss : f64,

    _w : na::DMatrix::<f64>,

    // Differential privacy
    _dp : bool,
    _epsilon : f64,
    _noise_scale : f64,
    _gradient_norm_bound : f64,
    _ms : dp::MeasurementDMatrix,

    // Classification or continuous value
    _classification : bool,

    // Debug
    _debug : bool
}



impl LogisticRegression {
    pub fn new(max_iter : usize, batch_size : usize, nfeat : usize, nclass : usize, eta : f64, mu : f64, classification: bool, debug : bool) -> LogisticRegression {
        let lr = LogisticRegression {

            _max_iter : max_iter,
            _batch_size : batch_size,
            _nfeat : nfeat,
            _nclass : nclass,
            _eta : eta,
            _mu : mu,
            _loss : 0.0,

            _w : na::DMatrix::from_fn(nfeat, nclass, |r,c| {rand::random::<f64>() - 0.5}),
            //_w : na::DMatrix::from_element(nfeat, nclass, 0.),

            _dp : false,
            _epsilon : 0.0,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,
            _ms : dp::MeasurementDMatrix::new(0.0),

            _classification : classification,

            _debug : debug,
        };

        return lr;
    }


    pub fn reset(&mut self) {
        self._w = na::DMatrix::from_element(self._nfeat, self._nclass, 0.);
        self._loss = 0.0;
        self._dp = false;
        self._batch_size = 1;

        println!();
        println!(" - Reset weights - - - - - - - - - - - - - -");
        println!();
    }


    pub fn enable_dp(&mut self, dp : bool, epsilon : f64, noise_scale : f64, gradient_norm_bound : f64) {
        self._dp = dp;
        self._epsilon = epsilon;
        self._noise_scale = noise_scale;
        self._gradient_norm_bound = gradient_norm_bound;

        self._ms.initialize(epsilon, noise_scale, gradient_norm_bound);
    }


    pub fn get_loss(&self) -> f64 {
        return self._loss;
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

        if self._dp {
            let grad1 = self._ms.invoke(grad.clone());
            return grad1;
        }

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


    pub fn fit(&mut self, x_train : Vec<Vec<f64>>, y_train : Vec<f64>, epochs1 : usize, batch1 : usize) {

        if self._dp {
            println!("Differential privacy enabled");
        }

        let mut li = na::DMatrix::from_element(x_train.len(), x_train[0].len(), 0.);
        //println!("fit");
        for i in 0..x_train.len() {
            let lii = na::DMatrix::<f64>::from_vec(1, x_train[0].len(), x_train[i].clone());
            for j in 0..lii.ncols() {
                li[(i,j)] = lii[(0,j)];
            }
        }

        let mut lo = na::DMatrix::from_element(x_train.len(), self._nclass, 0.);
        for i in 0..y_train.len() {
            let mut loi = na::DMatrix::from_element(1, self._nclass, 0.);
            loi[(0, y_train[i] as usize)] = 1.0;
            for j in 0..loi.ncols() {
                lo[(i,j)] = loi[(0,j)];
            }
        }

        let mut batch_idx : usize;
        let mut batch_size : usize;

        for i in 0..epochs1 {
            batch_idx = 0;
            batch_size = batch1;

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
                self._loss = self.loss(x_batch.clone(),y_batch.clone());

                if self._debug {
                    println!("Loss {:?} = {:.8}", i, self._loss);
                }
            }


            if x_train.len() - batch_idx > 0 {
                batch_size = x_train.len() - batch_idx;

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

                let grad = self.gradient(x_batch.clone(), y_batch.clone());
                self._w = self._w.clone() - self._eta * grad;
                self._loss = self.loss(x_batch.clone(),y_batch.clone());

                if self._debug {
                    println!("Loss {:?} = {:.8}", i, self._loss);
                }
            }

            //let gradient = self.gradient(li.clone(), lo.clone());
            //self._w = self._w.clone() - self._eta * gradient;
            //let loss = self.loss(li.clone(), lo.clone());
            //println!("Loss {:?} = {:.8}", i, loss);


            println!("Epoch {} / {} - Loss = {:.8}", i+1, epochs1, self._loss);
        }

        println!("Final loss    = {:.8}", self._loss);
        
    }


    pub fn test(&mut self, x_test : Vec<Vec<f64>>, y_test : Vec<f64>) {

        let mut correct : i64 = 0;

        for i in 0..x_test.len() {
            //if (i % 100) == 0 {
            //    println!("Testing = {:.2} %", 100.0 * i as f64 / x_test.len() as f64);
            //}
    
            let li = na::DMatrix::<f64>::from_vec(1, x_test[0].len(), x_test[i].clone());

            let z = - li * self._w.clone();
            //let a = z.map(|x| fact::softmax(x));
            let a = fact::softmax(z.clone());
            let maxid = self.get_max_idx(a.clone());


            if maxid == y_test[i] as i64 {
                correct = correct + 1;
            }
        }
    
        println!("Test accuracy = {} / {} = {}%\n", correct, x_test.len(), (correct as f64 / x_test.len() as f64) * 100.);
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

