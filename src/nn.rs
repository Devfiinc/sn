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
    _dims : (usize, usize),

    // Differential privacy
    _dp : bool,
    _epsilon : f64,
    _noise_scale : f64,
    _gradient_norm_bound : f64,
    _ms : dp::MeasurementDMatrix,

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
            _dims : (0, 0),

            _dp : false,
            _epsilon : 0.0,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,
            _ms : dp::MeasurementDMatrix::new(0.0),

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

            //MNIST
            //nn._model.push(nnlayer::NNLayer::new("reshape".to_string(),     vec![1, 1, 784],     vec![1, 28, 28],     "sigmoid".to_string(), learning_rate, false));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),      vec![1, 28, 28],     vec![5, 3, 1, 0],    "sigmoid".to_string(), learning_rate, false));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),      vec![5, 26, 26],     vec![5, 3, 1, 0],    "sigmoid".to_string(), learning_rate, false));
            nn._model.push(nnlayer::NNLayer::new("reshape".to_string(),     vec![5, 24, 24],     vec![1, 1, 5*24*24], "sigmoid".to_string(), learning_rate, false));
            nn._model.push(nnlayer::NNLayer::new("dense".to_string(),       vec![1, 5*24*24, 1], vec![1, 100, 1],     "sigmoid".to_string(), learning_rate, false));
            nn._model.push(nnlayer::NNLayer::new("dense".to_string(),       vec![1, 100, 1],     vec![1, 10, 1],      "softmax".to_string(), learning_rate, false));


            //let mut d1 : usize = 1;
            //let mut d2 : usize = 2 * d1.clone();
            //let mut d3 : usize = 2 * d2.clone();
            //let mut d4 : usize = 2 * d3.clone();
            //let mut d5 : usize = 2 * d4.clone();
            //let mut d6 : usize = 2 * d5.clone();

            //let mut s1 : usize = 96;        //96
            //let mut s2 : usize = s1 / 2;    //48
            //let mut s3 : usize = s2 / 2;    //24
            //let mut s4 : usize = s3 / 2;    //12
            //let mut s5 : usize = s4 / 2;    //6
            //let mut s6 : usize = s5 / 2;    //

            
            // Nerves tested in Python 
            /*
            nn._model.push(nnlayer::NNLayer::new("reshape".to_string(),     vec![1, 1, 96*96],     vec![1, 96, 96],     "sigmoid".to_string(), learning_rate, false));
            
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![ 1, 96, 96],     vec![d1, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 96, 96],     vec![d1, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d1, 96, 96],     vec![d1, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 48, 48],     vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 48, 48],     vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d2, 48, 48],     vec![d2, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2,  24, 24],    vec![d3, 3, 1, 0],   "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 24, 24],    vec![d3, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d3, 24, 24],    vec![d3, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 12, 12],    vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 12, 12],    vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d4, 12, 12],    vec![d4, 2, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 6, 6],      vec![d5, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d5, 6, 6],      vec![d5, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d5, 6, 6],      vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d4, 6, 6],      vec![d4, 2, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d4, 12, 12],    vec![d5, 12, 12, 12], "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d5, 12, 12],    vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 12, 12],    vec![d3, 3, 1, 0],    "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d3, 12, 12],    vec![d3, 2, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d3, 24, 24],    vec![d4, 24, 24, 9],   "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 24, 24],    vec![d3, 3, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 24, 24],    vec![d2, 3, 1, 0],     "relu".to_string(), learning_rate, true));

            nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d2, 24, 24],    vec![d2, 2, 1, 0],     "relu".to_string(), learning_rate, true));
            nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d2, 48, 48],    vec![d3, 48, 48, 6],   "relu".to_string(), learning_rate, true));
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
        }

        return nn;
    }


    // Construct NN
    pub fn _new_dense(layers : Vec<String>, topology : Vec<usize>, activation_functions : Vec<String>, learning_rate : f64, debug : bool) -> NN{
        let mut nn = NN {
            _layers : layers.clone(),
            _topology : topology.clone(),
            _activation_functions : activation_functions.clone(),
            _learning_rate : learning_rate,
            _model : vec![],
            _concatenations : vec![],
            _dims : (0, 0),

            _dp : false,
            _epsilon : 0.0,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,
            _ms : dp::MeasurementDMatrix::new(0.0),

            _debug : debug
        };

        let layer_type = layers[0].clone();

        for i in 0..topology.len() - 1 {
            let fact = activation_functions[i].clone();
            nn._model.push(nnlayer::NNLayer::new(layer_type.clone(), vec![1, topology[i], 1], vec![1, topology[i+1], 1], fact, learning_rate, false));    
            println!("--------------------------------------------------------------------------------");
            println!("  Layer {}: {} with topology {} - {}", i, layer_type.clone(), topology[i], topology[i+1]);
        }
        println!("--------------------------------------------------------------------------------");

        return nn;
    }
    


    // Construct NN
    pub fn _new_mnist(layers : Vec<String>, topology : Vec<usize>, activation_functions : Vec<String>, learning_rate : f64, debug : bool) -> NN{
        let mut nn = NN {
            _layers : layers.clone(),
            _topology : topology.clone(),
            _activation_functions : activation_functions.clone(),
            _learning_rate : learning_rate,
            _model : vec![],
            _concatenations : vec![],
            _dims : (0, 0),

            _dp : false,
            _epsilon : 0.0,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,
            _ms : dp::MeasurementDMatrix::new(0.0),

            _debug : debug
        };

        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),      vec![1, 28, 28],     vec![5, 3, 1, 0],    "sigmoid".to_string(), learning_rate, false));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),      vec![5, 26, 26],     vec![5, 3, 1, 0],    "sigmoid".to_string(), learning_rate, false));
        nn._model.push(nnlayer::NNLayer::new("reshape".to_string(),     vec![5, 24, 24],     vec![1, 1, 5*24*24], "sigmoid".to_string(), learning_rate, false));
        nn._model.push(nnlayer::NNLayer::new("dense".to_string(),       vec![1, 5*24*24, 1], vec![1, 100, 1],     "sigmoid".to_string(), learning_rate, false));
        nn._model.push(nnlayer::NNLayer::new("dense".to_string(),       vec![1, 100, 1],     vec![1, 10, 1],      "softmax".to_string(), learning_rate, false));

        for i in 0..nn._model.len() {
            println!("--------------------------------------------------------------------------------");
            println!("  Layer {}: {} with topology {:?} - {:?}", i, nn._model[i].get_layer_type(), nn._model[i].get_input_size(), nn._model[i].get_output_size());
            println!("--------------------------------------------------------------------------------");
        }

        return nn;
    }



    pub fn _new_nerves(layers : Vec<String>, topology : Vec<usize>, activation_functions : Vec<String>, learning_rate : f64, debug : bool) -> NN{
        let mut nn = NN {
            _layers : layers.clone(),
            _topology : topology.clone(),
            _activation_functions : activation_functions.clone(),
            _learning_rate : learning_rate,
            _model : vec![],
            _concatenations : vec![],
            _dims : (0, 0),

            _dp : false,
            _epsilon : 0.0,
            _noise_scale : 0.0,
            _gradient_norm_bound : 0.0,
            _ms : dp::MeasurementDMatrix::new(0.0),

            _debug : debug
        };

        let d1 : usize = 1;
        let d2 : usize = 2 * d1.clone();
        let d3 : usize = 2 * d2.clone();
        let d4 : usize = 2 * d3.clone();
        let d5 : usize = 2 * d4.clone();
        let d6 : usize = 2 * d5.clone();

        let s1 : usize = 96;        //96
        let s2 : usize = s1 / 2;    //48
        let s3 : usize = s2 / 2;    //24
        let s4 : usize = s3 / 2;    //12
        let s5 : usize = s4 / 2;    //6
        let s6 : usize = s5 / 2;    //

            
        // Nerves tested in Python 
        nn._model.push(nnlayer::NNLayer::new("reshape".to_string(),     vec![1, 1, 96*96],     vec![1, 96, 96],     "sigmoid".to_string(), learning_rate, false));
            
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![ 1, 96, 96],     vec![d1, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 96, 96],     vec![d1, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d1, 96, 96],     vec![d1, 2, 1, 0],    "relu".to_string(), learning_rate, true));

        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 48, 48],     vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 48, 48],     vec![d2, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d2, 48, 48],     vec![d2, 2, 1, 0],    "relu".to_string(), learning_rate, true));

        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2,  24, 24],    vec![d3, 3, 1, 0],   "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 24, 24],    vec![d3, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d3, 24, 24],    vec![d3, 2, 1, 0],    "relu".to_string(), learning_rate, true));

        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 12, 12],    vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 12, 12],    vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("max_pooling".to_string(),  vec![d4, 12, 12],    vec![d4, 2, 1, 0],    "relu".to_string(), learning_rate, true));

        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 6, 6],      vec![d5, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d5, 6, 6],      vec![d5, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d5, 6, 6],      vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));

        nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d4, 6, 6],      vec![d4, 2, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d4, 12, 12],    vec![d5, 12, 12, 12], "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d5, 12, 12],    vec![d4, 3, 1, 0],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 12, 12],    vec![d3, 3, 1, 0],    "relu".to_string(), learning_rate, true));

        nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d3, 12, 12],    vec![d3, 2, 1, 0],     "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d3, 24, 24],    vec![d4, 24, 24, 9],   "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d4, 24, 24],    vec![d3, 3, 1, 0],     "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 24, 24],    vec![d2, 3, 1, 0],     "relu".to_string(), learning_rate, true));

        nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d2, 24, 24],    vec![d2, 2, 1, 0],     "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d2, 48, 48],    vec![d3, 48, 48, 6],   "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d3, 48, 48],    vec![d2, 3, 1, 0],     "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 48, 48],    vec![d1, 3, 1, 0],     "relu".to_string(), learning_rate, true));

        nn._model.push(nnlayer::NNLayer::new("upsampling".to_string(),   vec![d1, 48, 48],    vec![d1, 2, 1, 0],     "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("concat".to_string(),       vec![d1, 96, 96],    vec![d2, 96, 96, 3],    "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d2, 96, 96],    vec![d1, 3, 1, 0],     "relu".to_string(), learning_rate, true));
        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 96, 96],    vec![d1, 3, 1, 0],     "relu".to_string(), learning_rate, true));

        nn._model.push(nnlayer::NNLayer::new("conv2d".to_string(),       vec![d1, 96, 96],     vec![1, 1, 1, 0],     "sigmoid".to_string(), learning_rate, true));

        for i in 0..nn._model.len() {
            println!("--------------------------------------------------------------------------------");
            println!("  Layer {}: {} with topology {:?} - {:?}", i, nn._model[i].get_layer_type(), nn._model[i].get_input_size(), nn._model[i].get_output_size());
            println!("--------------------------------------------------------------------------------");
        }

        return nn;
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


    // Forward propagation
    pub fn forward(&mut self, input : na::DMatrix::<f64>) -> na::DMatrix::<f64> {
        if self._debug {
            println!(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
            println!("     FORWARD PROPAGATION                                           ");
            println!(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
        }

        let mut vals : Vec<na::DMatrix::<f64>> = vec![input.transpose()];

        if self._debug {
            println!("Layer 0 : ");
            println!("        : out dims {} x {} x {}", vals.len(), vals[0].nrows(), vals[0].ncols());
        }

        for i in 0..self._model.len() {
            if self._debug {
                println!("Layer {} : {}", i+1, self._model[i].get_layer_type());
            }

            if self._model[i].get_layer_type() == "concat".to_string() {
                let pair = self._model[i].get_concat_pair();
                let prev_vals = self._model[pair].get_input_conv();
                vals = self._model[i].concat_forward(vals.clone(), prev_vals.clone());
            } else {
                vals = self._model[i].forward(vals);
            }

            if self._debug {
                println!("        : out dims {} x {} x {}", vals.len(), vals[0].nrows(), vals[0].ncols());
                for a in 0..vals[0].nrows() {
                    for b in 0..vals[0].ncols() {
                        print!("{:.2} ", vals[0][(a,b)]);
                    }
                    println!("");
                }
            }
        }

        return vals[0].clone();
    }



    // Back propagation
    pub fn backward(&mut self, output : na::DMatrix::<f64>, ytrain : na::DMatrix::<f64>) {

        if self._debug {
            println!(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
            println!("     BACK PROPAGATION                                              ");
            println!(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ");
        }

        let mut n : f64 = output.ncols() as f64;
        if output.nrows() > output.ncols() {
            n = output.nrows() as f64;
        } 

        let mut delta : Vec<na::DMatrix::<f64>> = vec![(2.0 / n) * (output.clone() - ytrain.clone())];

        if self._debug {
            println!("Layer {} : ", self._model.len());
            println!("        : out dims {} x {} x {}", delta.len(), delta[0].nrows(), delta[0].ncols());
        }

        for i in (0..self._model.len()).rev() {
            if self._debug {
                println!("Layer {} : {}", i, self._model[i].get_layer_type());
            }

            delta = self._model[i].backward(delta);

            if self._debug {
                println!("        : out dims {} x {} x {}", delta.len(), delta[0].nrows(), delta[0].ncols());
                for a in 0..delta[0].nrows() {
                    for b in 0..delta[0].ncols() {
                        print!("{:.2} ", delta[0][(a,b)]);
                    }
                    println!("");
                }
            }
        }

        //println!("Dice coefficient: {:.4}", self._model[0].dice_coef(ytrain.clone(), output.clone()));
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


    // Compute accuracy
    pub fn compute_accuracy_1(&mut self, x_val : Vec<Vec<f64>>, y_val : Vec<f64>) -> f64 {
        let mut v1 : usize = 0;
        let mut v0 : usize = 0;

        for i in 0..x_val.len() {
            if (i % 100) == 0 {
                println!("Validating = {:.2} %", 100.0 * i as f64 / x_val.len() as f64);
            }

            let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._dims.0, self._dims.1, x_val[i].clone());
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


    pub fn compute_accuracy_mdim_1(&mut self, x_val : Vec<Vec<f64>>, y_val : Vec<Vec<f64>>) -> f64 {
        let mut v1 : f64 = 0.0;
        let v0 : f64 = x_val.len() as f64;

        for i in 0..x_val.len() {
            if (i % 10) == 0 {
                println!(" - Validating = {:.2} %", 100.0 * i as f64 / x_val.len() as f64);
            }

            let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._dims.0, self._dims.1, x_val[i].clone());
            let lo : na::DMatrix::<f64> = self.forward(li.clone());
        
            let yi : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._dims.0, self._dims.1, y_val[i].clone());
            v1 += self._model[0].dice(yi, lo);
        }

        return 100.0 * (v1 / v0);
    }


    // Train reshaping input layer
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
            println!(" - Validation = {:.4} %", self.compute_accuracy(x_val.clone(), y_val.clone()));
            println!(" - - - - - - - - - - - - - - - - - - - - ");
            
        }
    }


    // Train without reshaping input layer
    pub fn train_1(&mut self, x_train : Vec<Vec<f64>>, y_train : Vec<f64>,
                            x_val : Vec<Vec<f64>>,   y_val : Vec<f64>, 
                            _num_episodes : i64, _max_steps : i64, _target_upd : i64, _exp_upd : i64,
                            dims : (usize, usize)) {

        self._dims = dims;

        for i in 0.._num_episodes {
            for i in 0..x_train.len() {
                if (i % 100) == 0 {
                    println!("Training = {:.2} %", 100.0 * i as f64 / x_train.len() as f64);
                }
                //println!("{:?} of {:?}", i+1, x_train.len());
        
                //println!("Forward");
                let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._dims.0, self._dims.1, x_train[i].clone());
                let lo : na::DMatrix::<f64> = self.forward(li.clone());
        
                //println!("Backward");
                //let mut y = na::DMatrix::from_element(1, self._topology[self._topology.len() - 1], 0.);
                let mut y = na::DMatrix::from_element(1, self._model[self._model.len() - 1].get_output_size()[1], 0.);
                y[(0, y_train[i].clone() as usize)] = 1.0;
        
                self.backward(lo.clone(), y.clone());
            }

            println!(" - ");
            println!(" - Validation = {:.4} %", self.compute_accuracy_1(x_val.clone(), y_val.clone()));
            println!(" - - - - - - - - - - - - - - - - - - - - ");
            
        }
    }


    // Train reshaping input layer
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
            println!(" - Validation = {:.4} %", self.compute_accuracy_mdim(x_val.clone(), y_val.clone()));
            println!(" - - - - - - - - - - - - - - - - - - - - ");
            
        }
    }


    // Train without reshaping input layer
    pub fn train_mdim_1(&mut self, x_train : Vec<Vec<f64>>, y_train : Vec<Vec<f64>>,
                                   x_val : Vec<Vec<f64>>,   y_val : Vec<Vec<f64>>, 
                                   _num_episodes : i64, _max_steps : i64, _target_upd : i64, _exp_upd : i64,
                                   dims : (usize, usize)) {

        self._dims = dims;

        for i in 0.._num_episodes {
            for i in 0..x_train.len() {
                if (i % 10) == 0 {
                    println!("Training = {:.2} %", 100.0 * i as f64 / x_train.len() as f64);
                }
                //println!("{:?} of {:?}", i+1, x_train.len());
        
                //println!("Forward");
                let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(dims.0, dims.1, x_train[i].clone());
                let lo : na::DMatrix::<f64> = self.forward(li.clone());
        
                //println!("Backward");
                //let mut y = na::DMatrix::from_element(1, self._topology[self._topology.len() - 1], 0.);
                //let mut y = na::DMatrix::from_element(1, self._model[self._model.len() - 1].get_output_size()[1], 0.);
                let mut y = na::DMatrix::<f64>::from_vec(dims.0, dims.1, y_train[i].clone());
                //y[(0, y_train[i].clone() as usize)] = 1.0;
        
                self.backward(lo.clone(), y.clone());
            }

            println!(" - ");
            println!(" - Validation = {:.4} %", self.compute_accuracy_mdim_1(x_val.clone(), y_val.clone()));
            println!(" - - - - - - - - - - - - - - - - - - - - ");
            
        }
    }


    // Train with 2D inputs and reshaping input layer
    pub fn train_mdim_2d(&mut self, x_train : Vec<Vec<Vec<f64>>>, y_train : Vec<Vec<Vec<f64>>>,
                            x_val : Vec<Vec<f64>>,   y_val : Vec<Vec<f64>>, 
                            _num_episodes : i64, _max_steps : i64, _target_upd : i64, _exp_upd : i64) {

        for i in 0.._num_episodes {
            for i in 0..x_train.len() {
                if (i % 100) == 0 {
                    println!("Training = {:.2} %", 100.0 * i as f64 / x_train.len() as f64);
                }
                //println!("{:?} of {:?}", i+1, x_train.len());
        
                //println!("Forward");
                let li : na::DMatrix::<f64> = na::DMatrix::<f64>::from_vec(self._topology[0], 1, x_train[i][0].clone());
                let lo : na::DMatrix::<f64> = self.forward(li.clone());
        
                //println!("Backward");
                //let mut y = na::DMatrix::from_element(1, self._topology[self._topology.len() - 1], 0.);
                //let mut y = na::DMatrix::from_element(1, self._model[self._model.len() - 1].get_output_size()[1], 0.);
                let mut y = na::DMatrix::<f64>::from_vec(self._topology[0], 1, y_train[i][0].clone());
                //y[(0, y_train[i].clone() as usize)] = 1.0;
        
                self.backward(lo.clone(), y.clone());
            }

            println!(" - ");
            println!(" - Validation = {:.4} %", self.compute_accuracy_mdim(x_val.clone(), y_val.clone()));
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