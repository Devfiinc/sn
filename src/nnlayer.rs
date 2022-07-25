
extern crate nalgebra as na;

use crate::fact;
use crate::dp;

pub struct NNLayer {
    // Layer definition
    _layer_type : String,

    _input_size : Vec<usize>,
    _input_size_depth : usize,
    _input_size_i : usize,
    _input_size_j : usize,

    _output_size : Vec<usize>,
    _output_size_depth : usize,
    _output_size_i : usize,
    _output_size_j : usize,
    _out_size : (usize, usize),

    _f_act : String,

    _learning_rate : f64,
    _stride : usize,
    _kern : usize,
    _padding : usize,
    _training : bool,

    // Dense Layer weights and values
    _weights : na::DMatrix::<f64>,
    _qvalues : na::DMatrix::<f64>,
    _qvaluesu : na::DMatrix::<f64>, 
    _input : na::DMatrix::<f64>,

    // Conv2d Layer weights and values
    _depth : usize,
    _kernels : Vec<Vec<na::DMatrix::<f64>>>,
    _num_kernels : usize,
    _qvalues_conv : Vec<na::DMatrix::<f64>>,
    _qvaluesu_conv : Vec<na::DMatrix::<f64>>,
    _input_conv : Vec<na::DMatrix::<f64>>,

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

    _smooth : f64,
    _full : bool
}


impl NNLayer {

    /*
    Dense layer 
        Input size      as per input layer
        Output size     as per output layer
        Layer type      "dense"

    Conv2D layer
        Input size      [depth, height, width]
        Output size     [kernel_size, num_of_kernels, stride, padding]
        Layer type      "conv2d"
    */

    // Construct NNLayer
    pub fn new(layer_type : String, input_size : Vec<usize>, output_size : Vec<usize>, f_act : String, learning_rate : f64, full : bool) -> NNLayer {
        let mut nnlayer = NNLayer {
            // Layer definition: Dense || Conv2d || MaxPooling || Dropout.
            _layer_type : layer_type.clone(),

            _input_size : input_size.clone(),
            _input_size_depth : input_size[0],
            _input_size_i : input_size[1],
            _input_size_j : input_size[2],

            _output_size : output_size.clone(),
            _output_size_depth : output_size[0],
            _output_size_i : output_size[1],
            _output_size_j : output_size[2],
            _out_size : (1, 1),

            _f_act : f_act.to_lowercase(),
            _learning_rate : learning_rate,


            // Dense layer
            _weights : na::DMatrix::from_fn(input_size[1], output_size[1], |_r,_c| {rand::random::<f64>() - 0.5}),
            _qvalues : na::DMatrix::from_element(1, input_size[1], 0.),
            _qvaluesu : na::DMatrix::from_element(1, input_size[1], 0.), 
            _input : na::DMatrix::from_element(1, input_size[1], 0.),


            // Conv2d layer
            _kern : output_size[0], // size of kernel
            _stride : output_size[1],
            _padding : output_size[2],
            _training : true,

            _depth : input_size[0],
            _num_kernels : output_size[2],  // number of kernels = number of output channels
            _input_conv : Vec::new(),
            _kernels : Vec::new(),
            _qvalues_conv : Vec::new(),
            _qvaluesu_conv : Vec::new(),


            // Differential Privacy
            _dp : false,
            _epsilon : 1.0,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,
            _ms : dp::MeasurementDMatrix::new(0.0),


            //Adam Optimizer
            _m : na::DMatrix::from_element(input_size[0], output_size[0], 0.), 
            _v : na::DMatrix::from_element(input_size[0], output_size[0], 0.), 
            _beta_1 : 0.9,
            _beta_2 : 0.999,
            _time : 1.0,
            _adam_epsilon : 0.00000001,

            _smooth : 1.0,
            _full : full,
        };

        if layer_type == "conv2d" {
            nnlayer._num_kernels = output_size[0];
            nnlayer._kern = output_size[1];
            nnlayer._stride = output_size[2];
            nnlayer._padding = output_size[3];

            nnlayer._depth = input_size[0];

            for i in 0..nnlayer._num_kernels {
                nnlayer._kernels.push(Vec::new());
                for j in 0..nnlayer._input_size_depth {
                    nnlayer._kernels[i].push(na::DMatrix::from_fn(nnlayer._kern, nnlayer._kern, |_r,_c| {rand::random::<f64>() - 0.5}));
                }
            }
        } 
        else if layer_type == "max_pooling" {
            nnlayer._num_kernels = output_size[0];
            nnlayer._kern = output_size[1];
            nnlayer._stride = output_size[2];
            nnlayer._padding = output_size[3];
            nnlayer._depth = input_size[0];
        }

        return nnlayer;
    }


    



    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Utils
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    pub fn get_layer_type(&self) -> String {
        return self._layer_type.clone();
    }

    pub fn get_input_size(&self) -> Vec<usize> {
        return self._input_size.clone();
    }

    pub fn get_output_size(&self) -> Vec<usize> {
        return self._output_size.clone();
    }









    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Differential Privacy controls
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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








    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Concatenate Layer
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    pub fn concat_forward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(input[i].clone());
        }

        for i in 0..self._qvalues_conv.len() {
            output.push(self._qvalues_conv[i].clone());
        }

        return output;
    }


    pub fn concat_backward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut output = Vec::new();
        for i in 0..input.len() {
            output.push(input[i].clone());
        }

        for i in 0..self._qvalues_conv.len() {
            output.push(self._qvalues_conv[i].clone());
        }

        return output;

    }



























    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Reshape Layer
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    pub fn flatten(&mut self, input : na::DMatrix::<f64>, column : bool) -> na::DMatrix::<f64> {
        let (rows, cols) = input.shape();

        let mut out = na::DMatrix::from_element(1, rows*cols, 0.);

        for a in 0..input.nrows() {
            for b in 0..input.ncols() {
                out[(1,a*cols + b)] = input[(a,b)];
            }
        }

        if column {
            return out.transpose();
        }
        return out;
    }


    pub fn reshape_forward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut output : Vec<na::DMatrix::<f64>> = Vec::new();
        
        let input_size_depth    = self._input_size_depth;
        let input_size_i        = self._input_size_i;
        let input_size_j        = self._input_size_j;

        let output_size_depth   = self._output_size_depth;
        let output_size_i       = self._output_size_i;
        let output_size_j       = self._output_size_j;

        // 1D to 2D
        if input_size_depth == output_size_depth {
            for im in 0..input.len() {
                let mut oi : usize = 0;
                let mut oj : usize = 0;
                let mut out = na::DMatrix::from_element(output_size_i, output_size_j, 0.);
                for a in 0..input[im].nrows() {
                    for b in 0..input[im].ncols() {
                        if oj == output_size_j {
                            oi += 1;
                            oj = 0;
                        }
                        out[(oi,oj)] = input[im][(a,b)];
                        oj += 1;
                    }
                }
                output.push(out);
            }
        } else if input_size_depth < output_size_depth {        // Put 1 vector into multiple classes

            let mut tmp : Vec<f64> = Vec::new();

            // Put all values into a signle vector
            for i in 0..input.len() {
                for a in 0..input[i].nrows() {
                    for b in 0..input[i].ncols() {
                        tmp.push(input[i][(a,b)]);
                    }
                }
            }

            // Reshape vector into matrix
            for i in 0..output_size_depth {
                output.push(na::DMatrix::from_element(output_size_i, output_size_j, 0.));
                for a in 0..output_size_i {
                    for b in 0..output_size_j {
                        output[i][(a,b)] = tmp[i*output_size_i*output_size_j + a*output_size_j + b];
                    }
                }
            }
        } else if input_size_depth > output_size_depth{         // Flatten multi into 1
            let mut tmp : Vec<f64> = Vec::new();

            // Put all values into a signle vector
            for i in 0..input.len() {
                for a in 0..input[i].nrows() {
                    for b in 0..input[i].ncols() {
                        tmp.push(input[i][(a,b)]);
                    }
                }
            }

            // Vector 1 row x n cols
            let mut out = na::DMatrix::from_element(1, tmp.len(), 0.);
            for i in 0..tmp.len() {
                out[(0,i)] = tmp[i];
            }

            // If output requires a column vector, push transpose
            if output_size_j == 1 {
                output.push(out.transpose());
            } else {
                output.push(out);
            }

        }

        return output;
    }



    pub fn reshape_backward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut output : Vec<na::DMatrix::<f64>> = Vec::new();
        
        let input_size_depth    = self._output_size_depth;
        let input_size_i        = self._output_size_i;
        let input_size_j        = self._output_size_j;

        let output_size_depth   = self._input_size_depth;
        let output_size_i       = self._input_size_i;
        let output_size_j       = self._input_size_j;

        // 1D to 2D
        if input_size_depth == output_size_depth {
            for im in 0..input.len() {
                let mut oi : usize = 0;
                let mut oj : usize = 0;
                let mut out = na::DMatrix::from_element(output_size_i, output_size_j, 0.);
                for a in 0..input[im].nrows() {
                    for b in 0..input[im].ncols() {
                        if oj == output_size_j {
                            oi += 1;
                            oj = 0;
                        }
                        out[(oi,oj)] = input[im][(a,b)];
                        oj += 1;
                    }
                }
                output.push(out);
            }
        } else if input_size_depth < output_size_depth {        // Put 1 vector into multiple classes

            let mut tmp : Vec<f64> = Vec::new();

            // Put all values into a signle vector
            for i in 0..input.len() {
                for a in 0..input[i].nrows() {
                    for b in 0..input[i].ncols() {
                        tmp.push(input[i][(a,b)]);
                    }
                }
            }

            // Reshape vector into matrix
            for i in 0..output_size_depth {
                output.push(na::DMatrix::from_element(output_size_i, output_size_j, 0.));
                for a in 0..output_size_i {
                    for b in 0..output_size_j {
                        output[i][(a,b)] = tmp[i*output_size_i*output_size_j + a*output_size_j + b];
                    }
                }
            }
        } else if input_size_depth > output_size_depth{         // Flatten multi into 1
            let mut tmp : Vec<f64> = Vec::new();

            // Put all values into a signle vector
            for i in 0..input.len() {
                for a in 0..input[i].nrows() {
                    for b in 0..input[i].ncols() {
                        tmp.push(input[i][(a,b)]);
                    }
                }
            }

            // Vector 1 row x n cols
            let mut out = na::DMatrix::from_element(1, tmp.len(), 0.);
            for i in 0..tmp.len() {
                out[(0,i)] = tmp[i];
            }

            // If output requires a column vector, push transpose
            if output_size_j == 1 {
                output.push(out.transpose());
            } else {
                output.push(out);
            }

        }

        return output;
    }











    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Transposed Convolutional Layer (ConvUp)
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    pub fn convolution_matrix(&mut self, m : na::DMatrix::<f64>, k : na::DMatrix::<f64>) -> na::DMatrix::<f64> {

        let (mr, mc) = m.shape();
        let (kr, kc) = k.shape();
        let (fr, fc) = self.conv2d_output_size(m.shape());

        let mut cmat = na::DMatrix::<f64>::from_element(fr*fc, mr+mc, 0.);

        let mut d1 : usize = 0;
        let mut d2 : usize = 0;
        for i in 0..cmat.nrows() {
            if i as f64 % fc as f64 == 0. && i > 0 {
                d1 += 1;
                d2 = 0;
            }
            for r in 0..kr {
                for c in 0..kc {
                    cmat[(i, d1*mc + d2 + r*mc + c)] = k[(r,c)];
                }
            }
        }
        return cmat;
    }


    pub fn convolution_matrix_kernel(&mut self, m : Vec<na::DMatrix::<f64>>, k : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut output : Vec<na::DMatrix::<f64>> = Vec::new();
        for i in 0..m.len() {
            output.push(self.convolution_matrix(m[i].clone(), k[i].clone()).transpose());
        }
        return output;
    }



    pub fn conv2d_up_forward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut output : Vec<na::DMatrix::<f64>> = Vec::new();

        let (fr, fc) = self.conv2d_output_size(input[0].shape());
        let (mr, mc) = input[0].shape();

        for k in 0..self._num_kernels {
            let conv_matrix = self.convolution_matrix_kernel(input.clone(), self._kernels[k].clone());
            
            let mut cmat = na::DMatrix::<f64>::from_element(fr*fc, 1, 0.);

            for i in 0..input.len() {
                let inp = input[i].clone().resize(mr*mc, 1, 0.);
                cmat = cmat + inp * conv_matrix[i].clone();
            }

            output.push(cmat);
        }

        return output;
    }



    pub fn conv2d_up_backward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut output : Vec<na::DMatrix::<f64>> = Vec::new();

        let (fr, fc) = self.conv2d_output_size(input[0].shape());
        let (mr, mc) = input[0].shape();

        for k in 0..self._num_kernels {
            let conv_matrix = self.convolution_matrix_kernel(input.clone(), self._kernels[k].clone());
            
            let mut cmat = na::DMatrix::<f64>::from_element(fr*fc, 1, 0.);

            for i in 0..input.len() {
                let inp = input[i].clone().resize(mr*mc, 1, 0.);
                cmat = cmat + inp * conv_matrix[i].clone();
            }

            output.push(cmat);
        }

        return output;
    }





    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Convolutional Layer
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    pub fn conv2d_output_size(&self, input_shape : (usize, usize)) -> (usize, usize) {
        let (input_rows, input_cols) = input_shape;
        let output_rows = ((input_rows as f32 - self._kern as f32 + 2.0 * self._padding as f32) / self._stride as f32) as usize + 1;
        let output_cols = ((input_cols as f32 - self._kern as f32 + 2.0 * self._padding as f32) / self._stride as f32) as usize + 1;
        return (output_rows, output_cols);
    }


    pub fn conv2d_forward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {

        self._input_conv = input.clone();

        self._out_size = self.conv2d_output_size(input[0].shape());


        

        println!("Full {}", self._full);
        if self._full {
            let add_size = (self._kern - 1) / 2;
            println!("Out size {} {}", self._out_size.0, self._out_size.1);
            self._out_size = self.conv2d_output_size((input[0].shape().0 + 2*add_size, input[0].shape().1 + 2*add_size));
            println!("Out size {} {}", self._out_size.0, self._out_size.1);

            for i in 0..input.len() {
                if add_size > 0 {
                    self._input_conv[i] = self._input_conv[i].clone().insert_rows(0, add_size, 0.0);
                    self._input_conv[i] = self._input_conv[i].clone().insert_columns(0, add_size, 0.0);
                    self._input_conv[i] = self._input_conv[i].clone().insert_rows(self._input_conv[i].nrows(), add_size, 0.0);
                    self._input_conv[i] = self._input_conv[i].clone().insert_columns(self._input_conv[i].ncols(), add_size, 0.0);
                }
            }
        }





        self._qvaluesu_conv = Vec::new();
        for i in 0..self._kernels.len() {
            self._qvaluesu_conv.push(na::DMatrix::from_element(self._out_size.0, self._out_size.1, 0.));
        }

        // For each element in the input matrix shape
        for i in (0..(self._input_size_i - self._kern + 1)).step_by(self._stride) {
            for j in (0..(self._input_size_j - self._kern + 1)).step_by(self._stride) {

                // For each element in the kernel shape
                for ki in 0..self._kern {
                    for kj in 0..self._kern {

                        // If the index is within the input matrix shape
                        if (i + ki) < self._input_conv[0].nrows() && (j + kj) < self._input_conv[0].ncols() {

                            // For each kernel
                            for k in 0..self._num_kernels {
                                // For each channel in the input image
                                for d in 0..self._input_size_depth {
                                    self._qvaluesu_conv[k][(i,j)] += self._input_conv[d][(i + ki, j + kj)] * self._kernels[k][d][(ki,kj)];
                                }
                            }
                        }
                    }
                }
            }
        }
      
        // Apply activation function to each element in each channel of the output matrix
        self._qvalues_conv = Vec::new();
        for i in 0..self._qvaluesu_conv.len() {
            if self._f_act == String::from("relu") {
                self._qvalues_conv.push(self._qvaluesu_conv[i].map(|x| fact::relu(x)));
            }
            else if self._f_act == String::from("sigmoid") {
                self._qvalues_conv.push(self._qvaluesu_conv[i].map(|x| fact::sigmoid(x)));
            }
            else if self._f_act == String::from("tanh") {
                self._qvalues_conv.push(self._qvaluesu_conv[i].map(|x| fact::tanh(x)));
            }
            else if self._f_act == String::from("softmax") {
                self._qvalues_conv.push(fact::softmax(self._qvaluesu_conv[i].clone()));
            }
            else {
                self._qvalues_conv.push(self._qvaluesu_conv[i].map(|x| fact::linear(x)));
            }
        }

        // Return the activated values
        return self._qvalues_conv.clone();
    }



    pub fn correlate2d(&self, sin1 : na::DMatrix::<f64>, sin2 : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        let mut in1 = sin1.clone();
        let mut in2 = sin2.clone();

        if in1.nrows() < in2.nrows() || in1.ncols() < in2.ncols() {
            println!("SWITCH INPUTS");
            in1 = sin2.clone();
            in2 = sin1.clone();
        }

        //let out_size = self.conv2d_output_size(in1.shape());
        let out_size = (self._kern, self._kern);
        let mut out = na::DMatrix::<f64>::from_element(out_size.0, out_size.1, 0.);

        for i in (0..(in1.nrows() - in2.nrows() + 1)).step_by(self._stride) {
            for j in (0..(in1.ncols() - in2.ncols()) + 1).step_by(self._stride) {

                // For each element in the kernel shape
                for ki in 0..in2.nrows() {
                    for kj in 0..in2.ncols() {

                        // If the index is within the input matrix shape
                        if (i + ki) < in1.nrows() && (j + kj) < in1.ncols() {

                            out[(i,j)] += in1[(i + ki, j + kj)] * in2[(ki,kj)];

                        }
                    }
                }
            }
        }

        return out;
    }


    pub fn rotate_matrix_180deg(&self, in1 : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        let mut out = na::DMatrix::<f64>::from_element(in1.nrows(), in1.ncols(), 0.);
        for i in 0..in1.nrows() {
            for j in 0..in1.ncols() {
                out[(i,j)] = in1[(in1.nrows() - i - 1, in1.ncols() - j - 1)];
            }
        }
        return out;
    }


    pub fn convolve2d(&self, sin1 : na::DMatrix::<f64>, sin2 : na::DMatrix::<f64>, full : bool) -> na::DMatrix::<f64> {
        let mut in1 = sin1.clone();
        let mut in2 = sin2.clone();

        if in1.nrows() < in2.nrows() || in1.ncols() < in2.ncols() {
            println!("SWITCH INPUTS");
            in1 = sin2.clone();
            in2 = sin1.clone();
        }

        in2 = self.rotate_matrix_180deg(in2.clone());
 

        let mut out_size = self.conv2d_output_size(in1.shape());
        if full {
            let mut add_size = self._kern - 1;
            if self._full {
                add_size /= 2;
            }
            out_size = self.conv2d_output_size((in1.shape().0 + 2*add_size, in1.shape().1 + 2*add_size));

            if add_size > 0 {
                in1 = in1.clone().insert_rows(0, add_size, 0.0);
                in1 = in1.clone().insert_columns(0, add_size, 0.0);
                in1 = in1.clone().insert_rows(in1.nrows(), add_size, 0.0);
                in1 = in1.clone().insert_columns(in1.ncols(), add_size, 0.0);
            }
            
        }

        let mut out = na::DMatrix::<f64>::from_element(out_size.0, out_size.1, 0.);

        for i in (0..(in1.nrows() - in2.nrows() + 1)).step_by(self._stride) {
            for j in (0..(in1.ncols() - in2.ncols()) + 1).step_by(self._stride) {

                // For each element in the kernel shape
                for ki in 0..in2.nrows() {
                    for kj in 0..in2.ncols() {

                        // If the index is within the input matrix shape
                        if (i + ki) < in1.nrows() && (j + kj) < in1.ncols() {

                            out[(i,j)] += in1[(i + ki, j + kj)] * in2[(ki,kj)];

                        }
                    }
                }
            }
        }

        return out;
    }


    pub fn conv2d_backward(&mut self, grad : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {

        let mut grad_kernels : Vec<Vec<na::DMatrix::<f64>>> = Vec::new();
        for i in 0..self._num_kernels {
            grad_kernels.push(Vec::new());
            for j in 0..self._input_size_depth {
                grad_kernels[i].push(na::DMatrix::from_element(self._kern, self._kern, 0.));
            }
        }

        let mut grad_input : Vec<na::DMatrix::<f64>> = Vec::new();
        for i in 0..self._input_size_depth {
            grad_input.push(na::DMatrix::from_element(self._input_size_i, self._input_size_j, 0.));
        }

        for i in 0..self._num_kernels {
            for j in 0..self._input_size_depth {
                grad_kernels[i][j] = self.correlate2d(self._input_conv[j].clone(), grad[i].clone());
                grad_input[j] += self.convolve2d(grad[i].clone(), self._kernels[i][j].clone(), true);
                break;
            }
        }

        for i in 0..self._num_kernels {
            for j in 0..self._input_size_depth {
                self._kernels[i][j] -= self._learning_rate * grad_kernels[i][j].clone();
            }
        }

        return grad_input;
    }










    // - - - - - - - - - - - - - - - - - - - - - - - -
    // Max Pooling Layer
    // - - - - - - - - - - - - - - - - - - - - - - - -

    pub fn max_pooling_output_size(&self, input_shape : (usize, usize)) -> (usize, usize) {
        let (input_rows, input_cols) = input_shape;

        let output_rows = ((input_rows as f64 - self._kern as f64).floor() / self._stride as f64) + 1.0;
        let output_cols = ((input_cols as f64 - self._kern as f64).floor() / self._stride as f64) + 1.0;

        return (output_rows as usize, output_cols as usize);
    }



    pub fn max_pooling_forward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        self._input_conv = input.clone();

        let out_size = self.max_pooling_output_size(input[0].shape());

        self._qvaluesu_conv = Vec::new();
        for i in 0..self._input_size_depth {
            self._qvaluesu_conv.push(na::DMatrix::from_element(out_size.0, out_size.1, 0.));
        }
        
        for i in (0..(self._input_size_i - self._kern + 1)).step_by(self._stride) {
            for j in (0..(self._input_size_j - self._kern + 1)).step_by(self._stride) {

                let mut max : Vec<f64> = Vec::new();
                for maxi in 0..self._input_size_depth {
                    max.push(0.);
                }

                for ki in 0..self._kern {
                    for kj in 0..self._kern {
                        for inp in 0..self._input_size_depth {
                            if self._input_conv[inp][(i + ki, j + kj)] > max[inp] {
                                max[inp] = self._input_conv[inp][(i + ki, j + kj)];
                            }
                        }
                    }
                }

                for inp in 0..self._input_size_depth {
                    self._qvaluesu_conv[inp][(i/self._stride,j/self._stride)] = max[inp];
                }
            }
        }
        return self._qvaluesu_conv.clone();
    }



    pub fn max_pooling_backward(&mut self, gradient_from_above : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut grad_input : Vec<na::DMatrix::<f64>> = Vec::new();
        for i in 0..self._input_conv.len() {
            grad_input.push(na::DMatrix::from_element(self._input_conv[i].nrows(), self._input_conv[i].ncols(), 0.));
        }
        
        for i in (0..(self._input_size_i - self._kern)).step_by(self._stride) {
            for j in (0..(self._input_size_j - self._kern)).step_by(self._stride) {

                let mut maxi : Vec<usize> = Vec::new();
                let mut maxj : Vec<usize> = Vec::new();
                let mut max : Vec<f64> = Vec::new();
                for h in 0..self._input_size_depth {
                    maxi.push(0);
                    maxj.push(0);
                    max.push(0.);
                }

                for ki in 0..self._kern {
                    for kj in 0..self._kern {
                        for inp in 0..self._input_size_depth {
                            if self._input_conv[inp][(i + ki, j + kj)] > max[inp] {
                                maxi[inp] = ki;
                                maxj[inp] = kj;
                                max[inp] = self._input_conv[inp][(i + ki, j + kj)];
                            }
                        }
                    }
                }

                for inp in 0..self._input_size_depth {
                    grad_input[inp][(i + maxi[inp], j + maxj[inp])] = max[inp];
                }
            }
        }
        return grad_input;
    }









    // - - - - - - - - - - - - - - - - - - - - - - - -
    // Dropout Layer
    // - - - - - - - - - - - - - - - - - - - - - - - -


    pub fn dropout_forward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {

        self._input = input[0].clone();

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
        
        return vec![self._qvalues.clone()];
    }



    pub fn dropout_backward(&mut self, gradient_from_above : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut adjusted_mul = gradient_from_above[0].clone();

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
            adjusted_mul[(0,i)] = qvalues_temp[(0,i)] * gradient_from_above[0][(0,i)];
        }

        let delta_i = &adjusted_mul * self._weights.transpose();

        let d_i = self._input.transpose() * adjusted_mul;

        self.update_weights(d_i.clone());

        return vec![delta_i];
    }















    // - - - - - - - - - - - - - - - - - - - - - - - -
    // Dense Layer
    // - - - - - - - - - - - - - - - - - - - - - - - -


    pub fn dense_forward(&mut self, input : na::DMatrix::<f64>) -> Vec<na::DMatrix::<f64>> {

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
        
        return vec![self._qvalues.clone()];
    }



    pub fn dense_backward(&mut self, gradient_from_above : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        let mut adjusted_mul = gradient_from_above[0].clone();

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
            adjusted_mul[(0,i)] = qvalues_temp[(0,i)] * gradient_from_above[0][(0,i)];
        }

        let delta_i = &adjusted_mul * self._weights.transpose();

        let d_i = self._input.transpose() * adjusted_mul;

        self.update_weights(d_i.clone());

        return vec![delta_i];
    }









    pub fn forward(&mut self, input : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        if self._layer_type == "dense".to_string() {
            return self.dense_forward(input[0].clone());
        }
        else if self._layer_type == "conv2d".to_string() {
            return self.conv2d_forward(input);
        }
        else if self._layer_type == "max_pooling".to_string() {
            return self.max_pooling_forward(input);
        }
        else if self._layer_type == "reshape".to_string() {
            return self.reshape_forward(input);
        }
        else if self._layer_type == "dropout".to_string() {
            return self.dropout_forward(input);
        }
        else if self._layer_type == "conv2dup".to_string() {
            return self.conv2d_up_forward(input);
        }
        else {
            return self.max_pooling_forward(input);
        }
    }

    pub fn backward(&mut self, gradient_from_above : Vec<na::DMatrix::<f64>>) -> Vec<na::DMatrix::<f64>> {
        if self._layer_type == "dense".to_string() {
            return self.dense_backward(gradient_from_above);
        }
        else if self._layer_type == "conv2d".to_string() {
            return self.conv2d_backward(gradient_from_above);
        }
        else if self._layer_type == "max_pooling".to_string() {
            return self.max_pooling_backward(gradient_from_above);
        }
        else if self._layer_type == "reshape".to_string() {
            return self.reshape_backward(gradient_from_above);
        }
        else if self._layer_type == "conv2dup".to_string() {
            return self.conv2d_up_backward(gradient_from_above);
        }
        else {
            return self.max_pooling_backward(gradient_from_above);
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

        let mut gradient2 = na::DMatrix::from_element(self._input_size[1], self._output_size[2], 0.);
        for i in 0..gradient1.nrows() {
            for j in 0..gradient1.ncols() {
                gradient2[(i,j)] = gradient1[(i,j)] * gradient1[(i,j)];
            }
        }

        v_temp = self._beta_2 * v_temp + (1.0 - self._beta_2) * gradient2.clone();
        // v_temp = self._beta_2 * v_temp + (1.0 - self._beta_2) * gradient.clone().powf(2.0);

        let m_vec_hat = &m_temp / (1.0 - self._beta_1.powf(self._time + 0.1));
        let v_vec_hat = &v_temp / (1.0 - self._beta_2.powf(self._time + 0.1));

        let mut weights_temp = na::DMatrix::from_element(self._input_size[1], self._output_size[1], 0.);
        
        for i in 0..self._weights.nrows() {
            for j in 0..self._weights.ncols() {
                weights_temp[(i,j)] = self._weights[(i,j)] - self._learning_rate * m_vec_hat[(i,j)] / (v_vec_hat[(i,j)] + self._adam_epsilon);
            }
        }

        self._m = m_temp.clone();
        self._v = v_temp.clone();
    }









    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Metric for medical data - Dice loss
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    pub fn dice_coef(&mut self, y_true : na::DMatrix<f64>, y_pred : na::DMatrix<f64>) -> f64 {
        let y_true_f = self.flatten(y_true.clone(), false);
        let y_pred_f = self.flatten(y_pred.clone(), false);
        let y_inter = y_true_f.clone() * y_pred_f.clone();

        let mut intersection : f64 = 0.0;
        let mut sum_y_true_f = 0.0;
        let mut sum_y_pred_f = 0.0;

        for i in 0..y_inter.nrows() {
            for j in 0..y_inter.ncols() {
                intersection += y_inter[(i,j)];
                sum_y_true_f += y_true_f[(i,j)];
                sum_y_pred_f += y_pred_f[(i,j)];
            }
        }

        return (2.0 * intersection + self._smooth) / (sum_y_true_f + sum_y_pred_f + self._smooth);
    }


    pub fn dice_coef_loss(&mut self, y_true : na::DMatrix<f64>, y_pred : na::DMatrix<f64>) -> f64 {
        return - 1.0 * self.dice_coef(y_true.clone(), y_pred.clone());
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
