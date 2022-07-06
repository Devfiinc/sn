use opendp::error::Fallible;
use opendp::trans::{make_split_lines, make_cast_default, make_clamp, make_bounded_sum};
use opendp::comb::{make_chain_tt, make_chain_mt};
use opendp::meas::*;

use opendp::core::*;
use opendp::dom::*;
use opendp::dist::*;
use opendp::meas::*;




extern crate nalgebra as na;
use na::{DMatrix, Hessenberg, Matrix4};

use rand::Rng;
type VecVec64 = Vec<Vec<Option<f64>>>;


use opendp::core::{Domain, Function, Measurement, PrivacyRelation, SensitivityMetric};
use opendp::dist::{AbsoluteDistance, L2Distance, SmoothedMaxDivergence};
use opendp::dom::{AllDomain, VectorDomain};
use opendp::traits::{CheckNull, TotalOrd};
use opendp::error::*;
use opendp::samplers::SampleGaussian;


use probability::*;


struct MeasurementDMatrix {
    pub _DI: DMatrix<f64>,
    pub _DO: DMatrix<f64>,
    pub _epsilon : f64,
}


impl MeasurementDMatrix {
    fn new(epsilon : f64) -> MeasurementDMatrix  {
        MeasurementDMatrix {
            _DI: DMatrix::<f64>::zeros(0, 0),
            _DO: DMatrix::<f64>::zeros(0, 0),
            _epsilon: epsilon,
        }
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


/*
    fn laplace(&self, epsilon: f64) -> f64 {
        let mut sum = 0.0;
        for i in 0..self._DI.nrows() {
            for j in 0..self._DI.ncols() {
                sum += self._DI[(i, j)] * self._DO[(i, j)] * self._DO[(i, j)];
            }
        }
        sum / (2.0 * epsilon)
    }
*/

    pub fn invoke(&mut self, DI: DMatrix<f64>) -> DMatrix<f64> {
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
