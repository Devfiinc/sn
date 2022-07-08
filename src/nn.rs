use crate::nnlayer;

extern crate nalgebra as na;



pub struct NN {
    // NN definition
    _topology : Vec<usize>,
    _activation_functions : Vec<String>,
    _learning_rate : f64,
    _model : Vec<nnlayer::NNLayer>,

    // Differential privacy
    _dp : bool,
    _noise_scale : f64,
    _gradient_norm_bound : f64,

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

            _dp : false,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,

            _debug : debug
        };

        for i in 0..topology.len() - 1 {
            let fact = activation_functions[i].clone();
            nn._model.push(nnlayer::NNLayer::new(topology[i] + 1, topology[i+1], fact, learning_rate, debug));            
        }

        return nn;
    }


    pub fn enable_dp(&mut self, dp : bool, noise_scale : f64, gradient_norm_bound : f64) {
        self._dp = dp;
        self._noise_scale = noise_scale;
        self._gradient_norm_bound = gradient_norm_bound;
    }


    pub fn disable_dp(&mut self) {
        self._dp = false;
    }


    // Forward propagation
    pub fn forward(&mut self, input : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        let mut vals = input.transpose().clone();

        //println!("Vals init {} x {}", vals.nrows(), vals.ncols());
        for i in 0..self._model.len() {
        //    println!("Layer {} of {} - vals {} x {}", i+1, self._model.len(), vals.nrows(), vals.ncols());
            vals = self._model[i].forward(vals.clone());
        }

        return vals;
    }



    // Back propagation
    pub fn backward(&mut self, output : na::DMatrix::<f64>, ytrain : na::DMatrix::<f64>) {
        let mut delta = output.clone() - ytrain.clone();

        //println!("Vals init {} x {}", delta.nrows(), delta.ncols());
        //for i in (0..self._model.len() - 1).rev() {
        for i in (0..self._model.len()).rev() {
            //println!("Layer {} of {} - vals {} x {}", i+1, self._model.len(), delta.nrows(), delta.ncols());
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
    pub fn train(&mut self, x_train : Vec<Vec<Option<f64>>>, y_train : Vec<Option<f64>>, _num_episodes : i64, _max_steps : i64, _target_upd : i64, _exp_upd : i64) {
        
        //let mut li = na::DMatrix::from_element(self._topology[0], 1, 0.);
        //let mut lo = na::DMatrix::from_element(self._topology[self._topology.len() - 1], 1, 0.);

        for i in 0..x_train.len() {
            if (i % 1000) == 0 {
                println!("Training = {:.2} %", 100.0 * i as f64 / x_train.len() as f64);
            }
    
            let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._topology[0], 1, x_train[i].iter().map(|x| x.unwrap()).collect());
            let lo : na::DMatrix::<f64> = self.forward(li.clone());
    
            let mut y = na::DMatrix::from_element(1, self._topology[self._topology.len() - 1], 0.);
            y[(0, y_train[i].unwrap() as usize)] = 1.0;
    
            self.backward(lo.clone(), y.clone());
        }
    }



    // Test
    pub fn test(&mut self, x_test : Vec<Vec<Option<f64>>>, y_test : Vec<Option<f64>>) {

        let mut correct : i64 = 0;

        for i in 0..x_test.len() {
            if (i % 100) == 0 {
                println!("Testing = {:.2} %", 100.0 * i as f64 / x_test.len() as f64);
            }
    
            let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._topology[0], 1, x_test[i].iter().map(|x| x.unwrap()).collect());
            let lo : na::DMatrix::<f64> = self.forward(li.clone());
    
            let mut maxid : usize = 0;
            let mut maxval : f64 = 0.0;
            for j in 0..lo.ncols(){
                if lo[(0,j)] > maxval {
                    maxid = j;
                    maxval = lo[(0,j)];
                }
            }
    
            if maxid == y_test[i].unwrap() as usize {
                correct = correct + 1;
            }
        }
    
        println!("Accuracy: {} / {} = {}%\n", correct, x_test.len(), (correct as f64 / x_test.len() as f64) * 100.);
    }

    




    // Predict
    pub fn predict(&mut self, input : Vec<Option<f64>>) -> i64{
        let li = na::DMatrix::<f64>::from_vec(input.len(), 1, input.iter().map(|x| x.unwrap()).collect());
        let lo = self.forward(li.clone());

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


    pub fn debug_mode(&mut self, debug : bool) {
        self._debug = debug;
    }

}