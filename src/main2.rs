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





/*
pub fn meas_DMatrix() -> Measurement<AllDomain<DMatrix<f64>>, AllDomain<DMatrix<f64>>,  SymmetricDistance, MaxDivergence<f64>> {
    Measurement::new(
        AllDomain::new(),
        AllDomain::new(),
        Function::new(|arg: &f64| arg.clone()),
        SymmetricDistance::default(),
        MaxDivergence::default(),
        PrivacyRelation::new(|_d_in, _d_out| true),
    )
}
*/



pub fn make_i32_identity() -> Transformation<AllDomain<i32>, AllDomain<i32>, AbsoluteDistance<i32>, AbsoluteDistance<i32>> {
    let input_domain = AllDomain::new();
    let output_domain = AllDomain::new();
    let function = Function::new(|arg: &i32| -> i32 { *arg });
    let input_metric = AbsoluteDistance::default();
    let output_metric = AbsoluteDistance::default();
    let stability_relation = StabilityRelation::new_from_constant(1);
    Transformation::new(input_domain, output_domain, function, input_metric, output_metric, stability_relation)
}



pub fn make_f64_identity() -> Transformation<AllDomain<f64>, AllDomain<f64>, AbsoluteDistance<f64>, AbsoluteDistance<f64>> {
    let input_domain = AllDomain::new();
    let output_domain = AllDomain::new();
    let function = Function::new(|arg: &f64| -> f64 { *arg });
    let input_metric = AbsoluteDistance::default();
    let output_metric = AbsoluteDistance::default();
    let stability_relation = StabilityRelation::new_from_constant(2.0);
    Transformation::new(input_domain, output_domain, function, input_metric, output_metric, stability_relation)
}



//pub fn meas_i32() -> Measurement<AllDomain<i32>, AllDomain<i32>, AbsoluteDistance<i32>, Measure::>

//DMatrix


/*
pub fn make_i32_identity_meas() -> Measurement<AllDomain<DMatrix>, AbsoluteDistance<i32>> {
    let domain = AllDomain::new();
    let metric = AbsoluteDistance::default();
    let function = Function::new(|arg: &i32| -> i32 { *arg });
    Measurement::new(domain, metric, function)
}

*/



fn print_DMatrix(D: &DMatrix<f64>) {
    for i in 0..D.nrows() {
        for j in 0..D.ncols() {
            print!("\t{}", D[(i, j)]);
        }
        println!("");
    }
}




fn main() {

    let tr1 = make_i32_identity();
    let in1 = 5;
    let out1 = tr1.invoke(&in1);
    println!("{}", out1.unwrap());

    //example().unwrap();

    let tr2 = make_f64_identity();
    let in2 = 10.0;
    let out2 = tr2.invoke(&in2);
    println!("{}", out2.unwrap());


    let dm1 = DMatrix::from_vec(2, 3, vec![1.0, 1.1, 1.2, 2.0, 2.1, 2.2]);




    let mut ms = MeasurementDMatrix::new(1.0);

    let dm2 = ms.invoke(dm1.clone());


    print_DMatrix(&dm1);
    print_DMatrix(&dm2);



}

