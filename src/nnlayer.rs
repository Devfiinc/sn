
extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

pub mod fact;



pub struct NNLayer {
    _input_size : i64,
    _output_size : i64,
    _learning_rate : f64,
    _f_act : String,

    _weights : na::DMatrix::<f64>,
    _qvalues : na::DMatrix::<f64>,
    _qvaluesu : na::DMatrix::<f64>, 
    _input : na::DMatrix::<f64>,

    //Adam Optimizer
    _m : na::DMatrix::<f64>, 
    _v : na::DMatrix::<f64>,
    _beta_1 : f64,
    _beta_2 : f64,
    _time : i64,
    _adam_epsilon : f64,

    _debug : bool
}


impl NNLayer {
    // Construct NNLayer
    pub fn new(input_size : i64, output_size : i64, f_act : String, learning_rate : f64, debug : bool) -> NNLayer {
        let mut nnlayer = NNLayer {
            _input_size : input_size,
            _output_size : output_size,
            _learning_rate : learning_rate,
            _f_act : f_act,
        
            _weights : na::DMatrix::from_fn(input_size, output_size, |r,c| {rand::random::<f64>() - 0.5}),
            _qvalues : na::DMatrix::from_element(input_size + 1, 1, 0.),
            _qvaluesu : na::DMatrix::from_element(input_size + 1, 1, 0.), 
            _input : na::DMatrix::from_element(input_size + 1, 1, 0.),
        
            //Adam Optimizer
            _m : na::DMatrix::from_element(input_size, output_size, 0.), 
            _v : na::DMatrix::from_element(input_size, output_size, 0.), 
            _beta_1 : 0.9,
            _beta_2 : 0.999,
            _time : 1,
            _adam_epsilon : 0.00000001,
        
            _debug : false
        }
    }

    pub fn debug_mode(&mut self, debug : bool) {
        self._debug = debug;
    }

    pub fn forward(&mut self, input : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        let mut input_with_bias = na::DMatrix::from_element(input_size + 1, 1, 0.);

        for i in 0..input.nrows() {
            input_with_bias[(i,0)] = input[(i,0)];
        }
        input_with_bias[(input.nrows(),0)] = 1.0;

        self._input = na::DMatrix::from_element(input_size + 1, 1, 0.);
        self._input = input_with_bias.clone();

        self._qvalues = input_with_bias * self._weights;
        self._qvaluesu = self._qvalues.clone();
    
    
        if _f_act == String::from("relu") {
            self._qvalues = self._qvalues.map(|x| fact::relu(x));
        }
        else if _f_act == String::from("sigmoid") {
            self._qvalues = self._qvalues.map(|x| fact::sigmoid(x));
        }
        else if _f_act == String::from("tanh") {
            self._qvalues = self._qvalues.map(|x| fact::tanh(x));
        }
        else {
            self._qvalues = self._qvalues.map(|x| fact::linear(x));
        }
        
        return self._qvalues;
    }


    pub fn update_weights(&mut self, gradient : na::DMatrix::<f64>) {
        let mut m_temp = na::DMatrix::from_element(self._input_size, self._output_size, 0.);
        let mut v_temp = na::DMatrix::from_element(self._input_size, self._output_size, 0.);
        let mut m_vec_hat = na::DMatrix::from_element(self._input_size, self._output_size, 0.);
        let mut v_vec_hat = na::DMatrix::from_element(self._input_size, self._output_size, 0.);

        m_temp = self._m.clone();
        v_temp = self._v.clone();

        m_temp = self._beta_1 * m_temp + (1.0 - self._beta_1) * gradient.clone();

        let gradient2 = na::DMatrix::from_element(self._input_size, self._output_size, 0.);
        for i in 0..gradient.nrows() {
            for j in 0..gradient.ncols() {
                gradient2[(i,j)] = gradient[(i,j)] * gradient[(i,j)];
            }
        }

        v_temp = self._beta_2 * v_temp + (1.0 - self._beta_2) * gradient2.clone();
        // v_temp = self._beta_2 * v_temp + (1.0 - self._beta_2) * gradient.clone().powf(2.0);

        m_vec_hat = m_temp / (1.0 - self._beta_1.powf(self._time + 0.1));
        v_vec_hat = v_temp / (1.0 - self._beta_2.powf(self._time + 0.1));

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
        let mut adjusted_mul = na::DMatrix::from_element(gradient_from_above.nrows(), 1, 0.);
        adjusted_mul = gradient_from_above.clone();

        let mut qvalues_temp = na::DMatrix::from_element(self._qvaluesu.nrows(), 1, 0.);
        qvalues_temp = self._qvaluesu.clone();


        if _f_act == String::from("relu") {
            self._qvalues = self._qvalues.map(|x| fact::relu_derivative(x));
        }
        else if _f_act == String::from("sigmoid") {
            self._qvalues = self._qvalues.map(|x| fact::sigmoid_derivative(x));
        }
        else if _f_act == String::from("tanh") {
            self._qvalues = self._qvalues.map(|x| fact::tanh_derivative(x));
        }
        else {
            self._qvalues = self._qvalues.map(|x| fact::linear_derivative(x));
        }

        for i in 0..adjusted_mul.nrows() {
            adjusted_mul[(i,0)] = qvalues_temp[(i,0)] * gradient_from_above[(i,0)];
        }

        let mut delta_i = na::DMatrix::from_element(self._input.nrows() - 1, 1, 0.);
        let mut delta_i_tmp = na::DMatrix::from_element(self._input.nrows(), 1, 0.);

        delta_i_tmp = adjusted_mul * self._weights.transpose();
        for i in 0..delta_i.nrows() {
            delta_i[(i,0)] = delta_i_tmp[(i,0)];
        }
        
        let mut D_i = na::DMatrix::from_element(self._input.nrows(), adjusted_mul.nrows(), 0.);
        for i in 0..self._input.nrows() {
            for j in 0..adjusted_mul.nrows() {
                D_i[(i,j)] = self._input[(i,0)] * adjusted_mul[(j,0)];
            }
        }

        self.update_weights(D_i.clone());

        return delta_i;
    }












    pub fn set_weights(&mut self, input : na::DMatrix::<f64>) {
        self._weights = input.clone();
    }


    pub fn get_weights(&self) -> na::DMatrix::<f64> {
        return self._weights.clone();
    }


    pub fn update_time(&mut self) {
        self._time = self._time + 1;
    }

}
