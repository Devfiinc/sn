use opendp::error::Fallible;
use opendp::trans::{make_split_lines, make_cast_default, make_clamp, make_bounded_sum};
use opendp::comb::{make_chain_tt, make_chain_mt};
use opendp::meas::*;

//use opendp::core::*;
//use opendp::dom::*;
//use opendp::dist::*;
//use opendp::meas::*;




extern crate nalgebra as na;
use na::{DMatrix, Hessenberg, Matrix4};

use rand::Rng;
type VecVec64 = Vec<Vec<Option<f64>>>;


//use opendp::core::{Domain, Function, Measurement, PrivacyRelation, SensitivityMetric};
//use opendp::dist::{AbsoluteDistance, L2Distance, SmoothedMaxDivergence};
//use opendp::dom::{AllDomain, VectorDomain};
//use opendp::traits::{CheckNull, TotalOrd};
//use opendp::error::*;
//use opendp::samplers::SampleGaussian;


use probability::*;

use std::cmp;


pub struct MeasurementDMatrix {
    pub _DI: DMatrix<f64>,
    pub _DO: DMatrix<f64>,
    pub _epsilon : f64,
    _bound_C : f64,
    _noise_scale : f64,
}


impl MeasurementDMatrix {
    pub fn new(epsilon : f64) -> MeasurementDMatrix  {
        MeasurementDMatrix {
            _DI: DMatrix::<f64>::zeros(0, 0),
            _DO: DMatrix::<f64>::zeros(0, 0),
            _epsilon: 0.0,
            _bound_C : 0.0,
            _noise_scale : 0.0,
        }
    }

    pub fn initialize(&mut self, epsilon: f64, bound_C : f64, noise_scale : f64) {
        self._epsilon = epsilon;
        self._bound_C = bound_C;
        self._noise_scale = noise_scale;
    }


    fn set_DI(&mut self, DI: DMatrix<f64>) {
        self._DI = DI;
    }

    fn set_DO(&mut self, DO: DMatrix<f64>) {
        self._DO = DO;
    }

    fn get_DI(&self) -> DMatrix<f64> {
        self._DI.clone()
    }

    fn get_DO(&self) -> DMatrix<f64> {
        self._DO.clone()
    }

    fn get_epsilon(&self) -> f64 {
        self._epsilon
    }


    fn laplace(&self, x : f64, mu: f64, b :f64) -> f64 {
        let xp = (x-mu).abs()/b;
        let laplace = (-xp).exp() / (2.0 * b);
        return laplace;
    }

    
    fn gaussian(&self, x : f64, mu: f64, b :f64) -> f64 {
        let xp = (x-mu).abs()/b;
        let gaussian = (-xp).exp() / (2.0 * b);
        return gaussian;
    }


    pub fn invoke(&mut self, DI: DMatrix<f64>) -> DMatrix<f64> {

        let mut DO = self.clip_gradient(DI.clone());
        DO = self.add_noise(DO.clone());

        return DO;
    }


    pub fn clip_gradient(&mut self, DI: DMatrix<f64>) -> DMatrix<f64> {
        let mut DO = DMatrix::<f64>::zeros(DI.nrows(), DI.ncols());
        let mut max_grad = 0.0;
        let mut max_grad_idx = 0;


        let mut norm : f64 = 0.0;
        for i in 0..DI.nrows() {
            for j in 0..DI.ncols() {
                norm = norm + DI[(i,j)]*DI[(i,j)];
            }
        }
        norm = norm.sqrt();

        let unit : f64 = 1.0;
        let normval : f64 = unit.max(norm/self._bound_C);

        for i in 0..DI.nrows() {
            for j in 0..DI.ncols() {
                DO[(i,j)] = DI[(i,j)]/normval;
            }
        }

        self._DO = DO.clone();
        return DO;
    }


    pub fn add_noise_laplace_2(&mut self, DI: DMatrix<f64>) -> DMatrix<f64> {
        self._DI = DI.clone();
        let s = self._DI.amax();
        let laplace = probability::distribution::Laplace::new(0.0, s / self._epsilon);

        self._DO = DI.clone();

        for i in 0..self._DI.nrows() {
            for j in 0..self._DI.ncols() {
                self._DO[(i,j)] = self.laplace(rand::random::<f64>(), 0.0, s / self._epsilon);
                //print!("\t{}", self._DI[(i, j)]);
            }
        }

        return self._DO.clone();
    }



    pub fn add_noise(&mut self, DI: DMatrix<f64>) -> DMatrix<f64> {
        self._DI = DI.clone();
        let s = self._DI.amax();
        let laplace = probability::distribution::Laplace::new(0.0, s / self._epsilon);

        let a = rand::random::<f64>();
        let b = self.laplace(a, 0.0, s / self._epsilon);
        println!("{} - {}", a, b);

        self._DO = DI.clone();

        for i in 0..self._DI.nrows() {
            for j in 0..self._DI.ncols() {
                self._DO[(i,j)] += self.laplace(rand::random::<f64>(), 0.0, s / self._epsilon);
                //print!("\t{}", self._DI[(i, j)]);
            }
        }

        return self._DO.clone();
    }
}
