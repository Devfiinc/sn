
extern crate nalgebra as na;

use crate::fact;
use crate::dp;

pub struct NNLayer {
    // Layer definition
    _input_size : usize,
    _output_size : usize,
    _layer_type : String,
    _f_act : String,
    _learning_rate : f64,

    // Convolution
    _stride : usize,
    _kern : usize,

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
    pub fn new(input_size : usize, output_size : usize, layer_type : String, f_act : String, learning_rate : f64, debug : bool) -> NNLayer {
        let nnlayer = NNLayer {
            _input_size : input_size,
            _output_size : output_size,
            _layer_type : layer_type,
            _f_act : f_act,
            _learning_rate : learning_rate,

            // Kernel
            _kern : input_size,
            _stride : output_size,

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
        
        //if layer_type == "Conv2D" {
        //    nnlayer._weights = na::DMatrix::from_fn(input_size, output_size, |_r,_c| {rand::random::<f64>() - 0.5});
        //}

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









    // - - - - - - - - - - - - - - - - - - - - - - - -
    // Convolutional Layer
    // - - - - - - - - - - - - - - - - - - - - - - - -

    pub fn conv2d_output_size(&self, input_shape : (usize, usize), stride : usize, padding : usize, kern : usize) -> (usize, usize) {
        let (input_rows, input_cols) = input_shape;
        let output_rows = ((input_rows - self._kern + 2 * padding) / self._stride) + 1;
        let output_cols = ((input_cols - self._kern + 2 * padding) / self._stride) + 1;
        return (output_rows, output_cols);
    }


    pub fn conv2d_forward(&mut self, input : na::DMatrix::<f64>) -> na::DMatrix::<f64> {

        self._input = input.clone();

        let out_size = self.conv2d_output_size(input.shape(), self._stride, 0, self._kern);
        self._qvaluesu = na::DMatrix::from_element(out_size.0, out_size.1, 0.);

        for i in (0..(self._input.nrows() - self._kern)).step_by(self._stride) {
            for j in (0..(self._input.ncols() - self._kern)).step_by(self._stride) {
                let mut out = 0.0;
                for ki in 0..self._kern {
                    for kj in 0..self._kern {
                        if (i + ki) < self._input.nrows() && (j + kj) < self._input.ncols() {
                            out += self._input[(i + ki, j + kj)] * self._weights[(ki, kj)];
                        }
                    }
                }
                self._qvaluesu[(i,j)] = out;
            }
        }

    
        if self._f_act == String::from("relu") {
            self._qvalues = self._qvaluesu.map(|x| fact::relu(x));
        }
        else if self._f_act == String::from("sigmoid") {
            self._qvalues = self._qvaluesu.map(|x| fact::sigmoid(x));
        }
        else if self._f_act == String::from("tanh") {
            self._qvalues = self._qvaluesu.map(|x| fact::tanh(x));
        }
        else if self._f_act == String::from("softmax") {
            self._qvalues = fact::softmax(self._qvaluesu.clone());
        }
        else {
            self._qvalues = self._qvaluesu.map(|x| fact::linear(x));
        }
        
        return self._qvalues.clone();
    }


    pub fn conv2d_backward(&mut self, grad : na::DMatrix::<f64>) -> na::DMatrix::<f64> {

        for i in (0..(self._input.nrows() - self._kern)).step_by(self._stride) {
            for j in (0..(self._input.ncols() - self._kern)).step_by(self._stride) {
                for ki in 0..self._kern {
                    for kj in 0..self._kern {
                        if (i + ki) < self._input.nrows() && (j + kj) < self._input.ncols() {
                            let gi = (i as f64 / self._stride as f64) as usize;
                            let gj = (j as f64 / self._stride as f64) as usize;
                            self._weights[(ki,kj)] -= self._learning_rate * grad[(gi,gj)] * self._input[(i+ki,j+kj)];
                        }
                    }
                }
                self._qvaluesu[(i,j)] = out;
            }
        }

        return delta_i;
    }






    // - - - - - - - - - - - - - - - - - - - - - - - -
    // Max Pooling Layer
    // - - - - - - - - - - - - - - - - - - - - - - - -

    pub fn max_pooling_output_size(&self, input_shape : (usize, usize), stride : usize, padding : usize) -> (usize, usize) {
        let (input_rows, input_cols) = input_shape;

        let output_rows = ((input_rows as f64 - self._kern as f64).floor() / self._stride as f64) + 1.0;
        let output_cols = ((input_cols as f64 - self._kern as f64).floor() / self._stride as f64) + 1.0;

        return (output_rows as usize, output_cols as usize);
    }



    pub fn max_pooling_forward(&mut self, input : na::DMatrix::<f64>, f : usize, s : usize) -> na::DMatrix::<f64> {
        self._input = input.clone();

        let out_size = self.max_pooling_output_size(input.shape(), f, s);
        self._qvaluesu = na::DMatrix::from_element(out_size.0, out_size.1, 0.);
        
        for i in (0..(self._input.nrows() - f)).step_by(s) {
            for j in (0..(self._input.ncols() - f)).step_by(s) {
                let mut max = 0.0;
                for ki in 0..f {
                    for kj in 0..f {
                        if (i + ki) < self._input.nrows() && (j + kj) < self._input.ncols() {
                            if self._input[(i + ki, j + kj)] > max {
                                max = self._input[(i + ki, j + kj)];
                            }
                        }
                    }
                }
                self._qvaluesu[(i,j)] = max;
            }
        }
        return self._qvaluesu.clone();
    }



    pub fn max_pooling_backward(&mut self, gradient_from_above : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
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

        for i in 0..adjusted_mul.ncols() {
            adjusted_mul[(0,i)] = qvalues_temp[(0,i)] * gradient_from_above[(0,i)];
        }

        let delta_i = &adjusted_mul * self._weights.transpose();

        let d_i = self._input.transpose() * adjusted_mul;

        self.update_weights(d_i.clone());

        return delta_i;
    }









    // - - - - - - - - - - - - - - - - - - - - - - - -
    // Dense Layer
    // - - - - - - - - - - - - - - - - - - - - - - - -


    pub fn dense_forward(&mut self, input : na::DMatrix::<f64>) -> na::DMatrix::<f64> {

        self._input = input.clone();

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
        }
        else {
            self._qvalues = self._qvalues.map(|x| fact::linear(x));
        }
        
        return self._qvalues.clone();
    }



    pub fn dense_backward(&mut self, gradient_from_above : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
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

        for i in 0..adjusted_mul.ncols() {
            adjusted_mul[(0,i)] = qvalues_temp[(0,i)] * gradient_from_above[(0,i)];
        }

        let delta_i = &adjusted_mul * self._weights.transpose();

        let d_i = self._input.transpose() * adjusted_mul;

        self.update_weights(d_i.clone());

        return delta_i;
    }









    pub fn forward(&mut self, input : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        if self._layer_type == "dense".to_string() {
            return self.dense_forward(input);
        }
        else if self._layer_type == "conv2d".to_string() {
            return self.conv2d_forward(input);
        }
        else if self._layer_type == "max_pooling".to_string() {
            return self.max_pooling_forward(input, self._kern, self._stride);
        }
        else {
            panic!("Unknown layer type");
        }
    }

    pub fn backward(&mut self, gradient_from_above : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        if self._layer_type == "dense".to_string() {
            return self.dense_backward(gradient_from_above);
        }
        else if self._layer_type == "conv2d".to_string() {
            return self.conv2d_backward(gradient_from_above);
        }
        else if self._layer_type == "max_pooling".to_string() {
            return self.max_pooling_backward(gradient_from_above);
        }
        else {
            panic!("Unknown layer type");
        }
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
