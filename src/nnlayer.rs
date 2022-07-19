
extern crate nalgebra as na;

use crate::fact;
use crate::dp;

pub struct NNLayer {
    // Layer definition
    _input_size : usize,
    _output_size : usize,
    _learning_rate : f64,
    _f_act : String,

    // Layer weights and values
    _weights : na::DMatrix::<f64>,
    _qvalues : na::DMatrix::<f64>,
    _qvaluesu : na::DMatrix::<f64>, 
    _input : na::DMatrix::<f64>,

    // Differential privacy
    _dp : bool,
    _epsilon : f64,
    _noise_scale : f64,
    _gradient_norm_bound : f64,
    _ms : dp::MeasurementDMatrix,

    //Adam Optimizer
    _m : na::DMatrix::<f64>, 
    _v : na::DMatrix::<f64>,
    _beta_1 : f64,
    _beta_2 : f64,
    _time : f64,
    _adam_epsilon : f64,

    _debug : bool
}


impl NNLayer {
    // Construct NNLayer
    pub fn new(input_size : usize, output_size : usize, f_act : String, learning_rate : f64, debug : bool) -> NNLayer {
        let nnlayer = NNLayer {
            _input_size : input_size,
            _output_size : output_size,
            _f_act : f_act,
            _learning_rate : learning_rate,

            _dp : false,
            _epsilon : 1.0,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,
            _ms : dp::MeasurementDMatrix::new(0.0),
        
            _weights : na::DMatrix::from_fn(input_size, output_size, |_r,_c| {rand::random::<f64>() - 0.5}),
            _qvalues : na::DMatrix::from_element(1, input_size /*+ 1*/, 0.),
            _qvaluesu : na::DMatrix::from_element(1, input_size /*+ 1*/, 0.), 
            _input : na::DMatrix::from_element(1, input_size /*+ 1*/, 0.),
        
            //Adam Optimizer
            _m : na::DMatrix::from_element(input_size, output_size, 0.), 
            _v : na::DMatrix::from_element(input_size, output_size, 0.), 
            _beta_1 : 0.9,
            _beta_2 : 0.999,
            _time : 1.0,
            _adam_epsilon : 0.00000001,
        
            _debug : debug
        };

        return nnlayer;
    }


    pub fn enable_dp(&mut self, dp : bool, epsilon : f64, noise_scale : f64, gradient_norm_bound : f64) {
        self._dp = dp;
        self._epsilon = epsilon;
        self._noise_scale = noise_scale;
        self._gradient_norm_bound = gradient_norm_bound;

        self._ms.initialize(epsilon, noise_scale, gradient_norm_bound);
    }


    pub fn disable_dp(&mut self) {
        self._dp = false;
    }


    pub fn debug_mode(&mut self, debug : bool) {
        self._debug = debug;
    }

    pub fn forward(&mut self, input : na::DMatrix::<f64>) -> na::DMatrix::<f64> {

        /*
        let mut input_with_bias = na::DMatrix::from_element(1, input.ncols() + 1, 0.);

        for i in 0..input.ncols() {
            input_with_bias[(0,i+1)] = input[(0,i)];
        }
        input_with_bias[(0,0)] = 1.0;

        self._input = input_with_bias.clone();
        */

        self._input = input.clone();

        //println!("input with bias {} {}", input_with_bias.nrows(), input_with_bias.transpose().ncols());
        //println!("weights          {} {}", self._weights.nrows(), self._weights.ncols());
        self._qvalues = self._input.clone() * self._weights.clone();
        self._qvaluesu = self._qvalues.clone();
    
        if self._f_act == String::from("relu") {
            self._qvalues = self._qvalues.map(|x| fact::relu(x));
        }
        else if self._f_act == String::from("sigmoid") {
            self._qvalues = self._qvalues.map(|x| fact::sigmoid(x));
        }
        else if self._f_act == String::from("tanh") {
            self._qvalues = self._qvalues.map(|x| fact::tanh(x));
        }
        else if self._f_act == String::from("softmax") {
            self._qvalues = fact::softmax(self._qvalues.clone());
            //self._qvalues = self._qvalues.map(|x| fact::softmax(x));
        }
        else {
            self._qvalues = self._qvalues.map(|x| fact::linear(x));
        }
        
        //println!("qvalues         {} {}", self._qvalues.nrows(), self._qvalues.ncols());
        return self._qvalues.clone();
    }


    pub fn update_weights(&mut self, gradient : na::DMatrix::<f64>) {
        self._weights = self._weights.clone() - self._learning_rate * gradient.clone();
    }


    pub fn update_weights_adam(&mut self, gradient : na::DMatrix::<f64>) {
        //let mut m_temp = na::DMatrix::from_element(self._input_size, self._output_size, 0.);
        //let mut v_temp = na::DMatrix::from_element(self._input_size, self._output_size, 0.);
        //let mut m_vec_hat = na::DMatrix::from_element(self._input_size, self._output_size, 0.);
        //let mut v_vec_hat = na::DMatrix::from_element(self._input_size, self._output_size, 0.);

        let mut m_temp = self._m.clone();
        let mut v_temp = self._v.clone();

        let mut gradient1 = gradient.clone();

        // Add privacy if dp mode is enabled
        if self._dp {
            gradient1 = self._ms.invoke(gradient.clone());
        }

        m_temp = self._beta_1 * m_temp + (1.0 - self._beta_1) * gradient1.clone();

        let mut gradient2 = na::DMatrix::from_element(self._input_size, self._output_size, 0.);
        for i in 0..gradient1.nrows() {
            for j in 0..gradient1.ncols() {
                gradient2[(i,j)] = gradient1[(i,j)] * gradient1[(i,j)];
            }
        }

        v_temp = self._beta_2 * v_temp + (1.0 - self._beta_2) * gradient2.clone();
        // v_temp = self._beta_2 * v_temp + (1.0 - self._beta_2) * gradient.clone().powf(2.0);

        let m_vec_hat = &m_temp / (1.0 - self._beta_1.powf(self._time + 0.1));
        let v_vec_hat = &v_temp / (1.0 - self._beta_2.powf(self._time + 0.1));

        let mut weights_temp = na::DMatrix::from_element(self._input_size, self._output_size, 0.);
        
        for i in 0..self._weights.nrows() {
            for j in 0..self._weights.ncols() {
                weights_temp[(i,j)] = self._weights[(i,j)] - self._learning_rate * m_vec_hat[(i,j)] / (v_vec_hat[(i,j)] + self._adam_epsilon);
            }
        }

        self._m = m_temp.clone();
        self._v = v_temp.clone();
    }


    pub fn backward(&mut self, gradient_from_above : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        let mut adjusted_mul = gradient_from_above.clone();

        let mut qvalues_temp = self._qvaluesu.clone();


        if self._f_act == String::from("relu") {
            qvalues_temp = qvalues_temp.map(|x| fact::relu_derivative(x));
        }
        else if self._f_act == String::from("sigmoid") {
            qvalues_temp = qvalues_temp.map(|x| fact::sigmoid_derivative(x));
        }
        else if self._f_act == String::from("tanh") {
            qvalues_temp = qvalues_temp.map(|x| fact::tanh_derivative(x));
        }
        else if self._f_act == String::from("softmax") {
            qvalues_temp = fact::softmax_derivative(qvalues_temp.clone());
            //self._qvalues = self._qvalues.map(|x| fact::softmax(x));
        }
        else {
            qvalues_temp = qvalues_temp.map(|x| fact::linear_derivative(x));
        }

        /*
        println!("qvalues_temp {:?}", qvalues_temp);
        println!("");

        println!("self._input shape {:?}", self._input.shape());
        println!("");

        println!("qvalues_temp shape {:?}", qvalues_temp.shape());
        println!("");

        println!("adjusted_mul shape {:?}", adjusted_mul.shape());
        println!("");

        println!("gradient_from_above shape {:?}", gradient_from_above.shape());
        println!("");

        println!("self._weights shape {:?}", self._weights.shape());
        println!("");
        */

        for i in 0..adjusted_mul.ncols() {
            adjusted_mul[(0,i)] = qvalues_temp[(0,i)] * gradient_from_above[(0,i)];
        }

        //println!("adjusted_mul {:?}", adjusted_mul);
        //println!("");

        let delta_i = &adjusted_mul * self._weights.transpose();

        //println!("delta_i {:?}", delta_i);
        //println!("");
        
        //println!("delta_i {} {}", delta_i.nrows(), delta_i.ncols());
        let d_i = self._input.transpose() * adjusted_mul;

        //let mut d_i = na::DMatrix::from_element(self._input.ncols(), adjusted_mul.ncols(), 0.);
        //for i in 0..self._input.nrows() {
        //    for j in 0..adjusted_mul.nrows() {
        //        d_i[(i,j)] = self._input[(i,0)] * adjusted_mul[(j,0)];
        //    }
        //}

        //println!("d_i {:?}", d_i);
        //println!("");

        self.update_weights(d_i.clone());

        return delta_i;
    }












    pub fn set_weights(&mut self, input : na::DMatrix::<f64>) {
        self._weights = input.clone();
    }


    pub fn get_weights(&self) -> na::DMatrix::<f64> {
        return self._weights.clone();
    }


    pub fn update_time(&mut self) {
        self._time = self._time + 1.0;
    }

}
