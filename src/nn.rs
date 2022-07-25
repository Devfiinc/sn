use crate::nnlayer;

extern crate nalgebra as na;



pub struct NN {
    // NN definition
    _layers : Vec<String>,
    _topology : Vec<usize>,
    _activation_functions : Vec<String>,
    _learning_rate : f64,
    _model : Vec<nnlayer::NNLayer>,
    _concatenations : Vec<usize>,

    // Differential privacy
    _dp : bool,
    _noise_scale : f64,
    _gradient_norm_bound : f64,

    _debug : bool
}

impl NN {
    // Construct NN
    pub fn new(layers : Vec<String>, topology : Vec<usize>, activation_functions : Vec<String>, learning_rate : f64, debug : bool) -> NN{
        let mut nn = NN {
            _layers : layers.clone(),
            _topology : topology.clone(),
            _activation_functions : activation_functions.clone(),
            _learning_rate : learning_rate,
            _model : vec![],
            _concatenations : vec![],

            _dp : false,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,

            _debug : debug
        };

        let layer_type = layers[0].clone();

        if layers[0] == "dense".to_string() {
            for i in 0..topology.len() - 1 {
                let fact = activation_functions[i].clone();
                nn._model.push(nnlayer::NNLayer::new(layer_type.clone(), vec![1, topology[i], 1], vec![1, topology[i+1], 1], fact, learning_rate, false));    
                println!("--------------------------------------------------------------------------------");
                println!("  Layer {}: {} with topology {} - {}", i, layer_type.clone(), topology[i], topology[i+1]);
            }
            println!("--------------------------------------------------------------------------------");
        } else {    //CNN
            //nn._model.push(nnlayer::NNLayer::new("reshape".to_string(),     vec![1, 1, 784],     vec![1, 28, 28],     "sigmoid".to_string(), learning_rate, false));
            //nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),      vec![1, 28, 28],     vec![5, 3, 1, 0],    "sigmoid".to_string(), learning_rate, true));
            //
            //nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(), vec![5, 28, 28], vec![5, 2, 2, 0],    "relu".to_string(), learning_rate, true));
            //nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),      vec![5, 14, 14],      vec![5, 2, 1, 0],    "sigmoid".to_string(), learning_rate, true));
            //
            //nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),    vec![5, 14, 14],     vec![5, 2, 1, 0],    "relu".to_string(), learning_rate, true));
            //nn._model.push(nnlayer::NNLayer::new("concat".to_string(),      vec![5, 28, 28],     vec![10, 28, 28, 2],    "relu".to_string(), learning_rate, true));
            //nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),      vec![10, 28, 28],    vec![5, 3, 2, 0],    "relu".to_string(), learning_rate, true));
            //
            //nn._model.push(nnlayer::NNLayer::new("reshape".to_string(),     vec![5, 28, 28],     vec![1, 1, 5*28*28], "sigmoid".to_string(), learning_rate, false));
            //nn._model.push(nnlayer::NNLayer::new("dense".to_string(),       vec![1, 5*28*28, 1], vec![1, 100, 1],     "sigmoid".to_string(), learning_rate, false));
            //nn._model.push(nnlayer::NNLayer::new("dense".to_string(),       vec![1, 100, 1],     vec![1, 10, 1],      "sigmoid".to_string(), learning_rate, false));


            let mut d1 : usize = 1;
            let mut d2 : usize = 2 * d1.clone();
            let mut d3 : usize = 2 * d2.clone();
            let mut d4 : usize = 2 * d3.clone();
            let mut d5 : usize = 2 * d4.clone();
            let mut d6 : usize = 2 * d5.clone();


            // Im size = 48 x 48
            nn._model.push(nnlayer::NNLayer::new("reshape".to_string(),     vec![1, 1, 48*48],     vec![1, 48, 48],     "sigmoid".to_string(), learning_rate, false));
                        
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![ 1, 48, 48],     vec![d1, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 48, 48],     vec![d1, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d1, 48, 48],     vec![d1, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 24, 24],     vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 24, 24],     vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d2, 24, 24],     vec![d2, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 12, 12],      vec![d3, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 12, 12],      vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d2, 12, 12],    vec![d2, 2, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d2, 24, 24],    vec![d3, 24, 24, 6],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 24, 24],    vec![d2, 3, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 24, 24],    vec![d1, 3, 1, 0],     "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d1, 24, 24],    vec![d1, 2, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d1, 48, 48],    vec![d2, 48, 48, 3],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 48, 48],    vec![d1, 3, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 48, 48],    vec![d1, 3, 1, 0],     "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 48, 48],     vec![1, 1, 1, 0],     "sigmoid".to_string(), learning_rate, true));





/*
            // Im size = 96 x 96
            nn._model.push(nnlayer::NNLayer::new("reshape".to_string(),     vec![1, 1, 96*96],     vec![1, 96, 96],     "sigmoid".to_string(), learning_rate, false));
            
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![ 1, 96, 96],     vec![d1, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 96, 96],     vec![d1, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d1, 96, 96],     vec![d1, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 48, 48],     vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 48, 48],     vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d2, 48, 48],     vec![d2, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2,  24, 24],    vec![d3, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 24, 24],    vec![d3, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d3, 24, 24],    vec![d3, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 12, 12],    vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 12, 12],    vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d4, 12, 12],    vec![d4, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 6, 6],      vec![d5, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d5, 6, 6],      vec![d5, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d5, 6, 6],      vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d4, 6, 6],      vec![d4, 2, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d4, 12, 12],    vec![d5, 12, 12, 12],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d5, 12, 12],    vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 12, 12],    vec![d3, 3, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d3, 12, 12],    vec![d3, 2, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d3, 24, 24],    vec![d4, 24, 24, 9],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 24, 24],    vec![d3, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 24, 24],    vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d2, 24, 24],    vec![d2, 2, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d2, 48, 48],    vec![d3, 48, 48, 6],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 48, 48],    vec![d2, 3, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 48, 48],    vec![d1, 3, 1, 0],     "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d1, 48, 48],    vec![d1, 2, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d1, 96, 96],    vec![d2, 96, 96, 3],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 96, 96],    vec![d1, 3, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 96, 96],    vec![d1, 3, 1, 0],     "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 96, 96],     vec![1, 1, 1, 0],     "sigmoid".to_string(), learning_rate, true));
*/


            for i in 0..nn._model.len() {
                println!("--------------------------------------------------------------------------------");
                println!("  Layer {}: {} with topology {:?} - {:?}", i, nn._model[i].get_layer_type(), nn._model[i].get_input_size(), nn._model[i].get_output_size());
                println!("--------------------------------------------------------------------------------");
            }

            println!("--------------------------------------------------------------------------------");
            println!("{} {} {} {} {} {}", d1, d2, d3, d4, d5, d6);
            println!("--------------------------------------------------------------------------------");
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
        //println!(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
        //println!("     FORWARD PROPAGATION                                           ");
        //println!(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");

        let mut vals : Vec<na::DMatrix::<f64>> = vec![input.transpose().clone()];

        //println!("Layer 0 : ");
        //println!("        : out dims {} x {} x {}", vals.len(), vals[0].nrows(), vals[0].ncols());

        for i in 0..self._model.len() {
            //println!("Layer {} : {}", i+1, self._model[i].get_layer_type());
            if self._model[i].get_layer_type() == "concat".to_string() {
                let pair = self._model[i].get_concat_pair();
                let prev_vals = self._model[pair].get_input_conv();
                vals = self._model[i].concat_forward(vals.clone(), prev_vals.clone());
            } else {
                vals = self._model[i].forward(vals);
            }

            //println!("        : out dims {} x {} x {}", vals.len(), vals[0].nrows(), vals[0].ncols());
            //println!("global forward");
            //for a in 0..vals[0].nrows() {
            //    for b in 0..vals[0].ncols() {
            //        print!("{:.2} ", vals[0][(a,b)]);
            //    }
            //    println!("");
            //}

        }

        return vals[0].clone();
    }



    // Back propagation
    pub fn backward(&mut self, output : na::DMatrix::<f64>, ytrain : na::DMatrix::<f64>) {
        //println!(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
        //println!("     BACK PROPAGATION                                              ");
        //println!(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");

        let mut n : f64 = output.ncols() as f64;
        if output.nrows() > output.ncols() {
            n = output.nrows() as f64;
        } 
        //let mut delta : Vec<na::DMatrix::<f64>> = vec![(2.0 / n) * (output.clone() - ytrain.clone())];

        let ytr = self._model[0].reshape(ytrain.clone(), (48,48));
        let mut delta : Vec<na::DMatrix::<f64>> = vec![(2.0 / n) * (output.clone() - ytr.clone())];
        
        //println!("Layer {} : ", self._model.len());
        //println!("        : out dims {} x {} x {}", delta.len(), delta[0].nrows(), delta[0].ncols());

        for i in (0..self._model.len()).rev() {
            //println!("Layer {} : {}", i, self._model[i].get_layer_type());
            delta = self._model[i].backward(delta.clone());

            //println!("        : out dims {} x {} x {}", delta.len(), delta[0].nrows(), delta[0].ncols());
            //println!("global backward");
            //for a in 0..delta[0].nrows() {
            //    for b in 0..delta[0].ncols() {
            //        print!("{:.2} ", delta[0][(a,b)]);
            //    }
            //    println!("");
            //}
        }

        println!("Dice coefficient: {:.4}", self._model[0].dice_coef(output.clone(), ytr.clone()));
    }



    // Update time
    pub fn update_time(&mut self) {
        for i in 0..self._model.len() {
            self._model[i].update_time();
        }
    }



    // Possition of max
    pub fn argmax(&self, v : na::DMatrix::<f64>) -> f64 {
        let mut maxval : f64 = 0.0;
        let mut maxidx : f64 = 0.0;
        for i in 0..v.nrows() {
            for j in 0..v.ncols() {
                if v[(i,j)] > maxval {
                    maxval = v[(i,j)];
                    if v.nrows() > v.ncols() {
                        maxidx = i as f64;
                    } else {
                        maxidx = j as f64;
                    }
                }
            }
        }
        return maxidx;
    }


    // Compute accuracy
    pub fn compute_accuracy(&mut self, x_val : Vec<Vec<f64>>, y_val : Vec<f64>) -> f64 {
        let mut v1 : usize = 0;
        let mut v0 : usize = 0;

        for i in 0..x_val.len() {
            if (i % 100) == 0 {
                println!("Validating = {:.2} %", 100.0 * i as f64 / x_val.len() as f64);
            }

            let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._topology[0], 1, x_val[i].clone());
            let lo : na::DMatrix::<f64> = self.forward(li.clone());
        
            let yo = self.argmax(lo);
            if yo == y_val[i] {
                v1 += 1;
            }
            v0 += 1;            
        }

        return 100.0 * (v1 as f64 / v0 as f64);
    }


    pub fn compute_accuracy_mdim(&mut self, x_val : Vec<Vec<f64>>, y_val : Vec<Vec<f64>>) -> f64 {
        let mut v1 : usize = 0;
        let mut v0 : usize = 0;

        for i in 0..x_val.len() {
            if (i % 100) == 0 {
                println!("Validating = {:.2} %", 100.0 * i as f64 / x_val.len() as f64);
            }

            let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._topology[0], 1, x_val[i].clone());
            let lo : na::DMatrix::<f64> = self.forward(li.clone());
        
            let yo = self.argmax(lo);
            if yo == 1.0 { //y_val[i] {
                v1 += 1;
            }
            v0 += 1;            
        }

        return 100.0 * (v1 as f64 / v0 as f64);
    }


    // Train
    pub fn train(&mut self, x_train : Vec<Vec<f64>>, y_train : Vec<f64>,
                            x_val : Vec<Vec<f64>>,   y_val : Vec<f64>, 
                            _num_episodes : i64, _max_steps : i64, _target_upd : i64, _exp_upd : i64) {

        for i in 0.._num_episodes {
            for i in 0..x_train.len() {
                if (i % 100) == 0 {
                    println!("Training = {:.2} %", 100.0 * i as f64 / x_train.len() as f64);
                }
                //println!("{:?} of {:?}", i+1, x_train.len());
        
                //println!("Forward");
                let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._topology[0], 1, x_train[i].clone());
                let lo : na::DMatrix::<f64> = self.forward(li.clone());
        
                //println!("Backward");
                //let mut y = na::DMatrix::from_element(1, self._topology[self._topology.len() - 1], 0.);
                let mut y = na::DMatrix::from_element(1, self._model[self._model.len() - 1].get_output_size()[1], 0.);
                y[(0, y_train[i].clone() as usize)] = 1.0;
        
                self.backward(lo.clone(), y.clone());
            }

            println!(" - ");
            println!(" - Validation = {:.2} %", self.compute_accuracy(x_val.clone(), y_val.clone()));
            println!(" - - - - - - - - - - - - - - - - - - - - ");
            
        }
    }


    // Train
    pub fn train_mdim(&mut self, x_train : Vec<Vec<f64>>, y_train : Vec<Vec<f64>>,
                            x_val : Vec<Vec<f64>>,   y_val : Vec<Vec<f64>>, 
                            _num_episodes : i64, _max_steps : i64, _target_upd : i64, _exp_upd : i64) {

        for i in 0.._num_episodes {
            for i in 0..x_train.len() {
                if (i % 100) == 0 {
                    println!("Training = {:.2} %", 100.0 * i as f64 / x_train.len() as f64);
                }
                //println!("{:?} of {:?}", i+1, x_train.len());
        
                //println!("Forward");
                let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._topology[0], 1, x_train[i].clone());
                let lo : na::DMatrix::<f64> = self.forward(li.clone());
        
                //println!("Backward");
                //let mut y = na::DMatrix::from_element(1, self._topology[self._topology.len() - 1], 0.);
                //let mut y = na::DMatrix::from_element(1, self._model[self._model.len() - 1].get_output_size()[1], 0.);
                let mut y = na::DMatrix::<f64>::from_vec(self._topology[0], 1, y_train[i].clone());
                //y[(0, y_train[i].clone() as usize)] = 1.0;
        
                self.backward(lo.clone(), y.clone());
            }

            println!(" - ");
            println!(" - Validation = {:.2} %", self.compute_accuracy_mdim(x_val.clone(), y_val.clone()));
            println!(" - - - - - - - - - - - - - - - - - - - - ");
            
        }
    }



    // Test
    pub fn test(&mut self, x_test : Vec<Vec<f64>>, y_test : Vec<f64>) {

        let mut correct : i64 = 0;

        for i in 0..x_test.len() {
            if (i % 100) == 0 {
                println!("Testing = {:.2} %", 100.0 * i as f64 / x_test.len() as f64);
            }
    
            let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._topology[0], 1, x_test[i].clone());
            let lo : na::DMatrix::<f64> = self.forward(li.clone());
    
            let mut maxid : usize = 0;
            let mut maxval : f64 = 0.0;
            for j in 0..lo.ncols(){
                if lo[(0,j)] > maxval {
                    maxid = j;
                    maxval = lo[(0,j)];
                }
            }
    
            if maxid == y_test[i].clone() as usize {
                correct = correct + 1;
            }
        }
    
        println!("Accuracy: {} / {} = {}%\n", correct, x_test.len(), (correct as f64 / x_test.len() as f64) * 100.);
    }

    




    // Predict
    pub fn predict(&mut self, input : Vec<f64>) -> i64{

        let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._topology[0], 1, input.clone());
        let lo : na::DMatrix::<f64> = self.forward(li.clone());

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