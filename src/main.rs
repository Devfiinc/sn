
use postgres::{Client, Error, NoTls};
use ndarray::*;

extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

mod fact;
mod dp;
mod lr;
mod nnlayer;
mod nn;
//pub mod nn;
//
//mod fact;
//mod nnlayer;
//mod nn;


type VecVec64 = Vec<Vec<Option<f64>>>;


use rand::thread_rng;
use rand::seq::SliceRandom;

//use opendp::*;
use opendp::core::*;
use opendp::meas::{make_base_geometric};

use opendp::core::{Domain, Function, Measure, Measurement, Metric, PrivacyRelation};
use opendp::dom::AllDomain;
use opendp::error::*;
use opendp::traits::{CheckNull, InfSub};


//use opendp::trans::{make_identity};
//use opendp::dom::*;//{VectorDomain, AllDomain};
//use opendp::dist::{SymmetricDistance};

//use opendp::core::{Function, Measurement, PrivacyRelation};
//use opendp::dist::{IntDistance, L1Distance, SmoothedMaxDivergence};
//use opendp::dom::{AllDomain, MapDomain};
//use opendp::error::Fallible;
//use opendp::samplers::SampleLaplace;
//use opendp::traits::{CheckNull, InfCast};




fn main() -> Result<(), Error> {

    // Read from Postgres database, same as spi does from within PGX
    // Data into VecVec<Option<f64>>>

    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();

    let mut sn : VecVec64 = vec![];

    for row in conn.query("SELECT * from iris", &[])? {

        sn.push(vec![row.get(0),
                     row.get(1),
                     row.get(2),
                     row.get(3),
                     row.get(4)]);
    }
    
    
    // Shuffle input
    sn.shuffle(&mut thread_rng());


    // Split dataset into train, cross validation and test
    let mut x_train : VecVec64 = vec![];
    let mut y_train : Vec<Option<f64>> = vec![];
    let mut x_cv : VecVec64 = vec![];
    let mut y_cv : Vec<Option<f64>> = vec![];
    let mut x_test : VecVec64 = vec![];
    let mut y_test : Vec<Option<f64>> = vec![];

    let mut size_train = 0.60;
    let mut size_cv = 0.20;
    let mut size_test = 0.20;

    let mut i : i64 = 0;
    let split_train : i64 = (sn.len() as f64 * size_train) as i64;
    let split_cv : i64 = (sn.len() as f64 * (size_train + size_cv)) as i64;
    for n in sn {
        if i < split_train {
            x_train.push(n[0..20].to_vec());
            y_train.push(n[20].clone());
        } else if i < split_cv {
            x_cv.push(n[0..20].to_vec());
            y_cv.push(n[20].clone());
        } else {
            x_test.push(n[0..20].to_vec());
            y_test.push(n[20].clone());
        }
        i = i + 1;
    }







    let mut lr = lr::LogisticRegression::new(1000, 0.1, 0.01, false);

    lr.fit(x_train.clone(), y_train.clone());

    lr.test(x_train.clone(), y_train.clone());




    /*


    

    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();

    let mut sn : VecVec64 = vec![];

    for row in conn.query("SELECT * from sn", &[])? {

        sn.push(vec![row.get(0),
                     row.get(1),
                     row.get(2),
                     row.get(3),
                     row.get(4),
                     row.get(5),
                     row.get(6),
                     row.get(7),
                     row.get(8),
                     row.get(9),
                     row.get(10),
                     row.get(11),
                     row.get(12),
                     row.get(13),
                     row.get(14),
                     row.get(15),
                     row.get(16),
                     row.get(17),
                     row.get(18),
                     row.get(19),
                     row.get(20)]);
    }
    
    
    // Shuffle input
    sn.shuffle(&mut thread_rng());


    // Split dataset into train, cross validation and test
    let mut x_train : VecVec64 = vec![];
    let mut y_train : Vec<Option<f64>> = vec![];
    let mut x_cv : VecVec64 = vec![];
    let mut y_cv : Vec<Option<f64>> = vec![];
    let mut x_test : VecVec64 = vec![];
    let mut y_test : Vec<Option<f64>> = vec![];

    let mut size_train = 0.60;
    let mut size_cv = 0.20;
    let mut size_test = 0.20;

    let mut i : i64 = 0;
    let split_train : i64 = (sn.len() as f64 * size_train) as i64;
    let split_cv : i64 = (sn.len() as f64 * (size_train + size_cv)) as i64;
    for n in sn {
        if i < split_train {
            x_train.push(n[0..20].to_vec());
            y_train.push(n[20].clone());
        } else if i < split_cv {
            x_cv.push(n[0..20].to_vec());
            y_cv.push(n[20].clone());
        } else {
            x_test.push(n[0..20].to_vec());
            y_test.push(n[20].clone());
        }
        i = i + 1;
    }


    let size_li = x_train[0].len();
    let size_l1 = 50;
    let size_l2 = 25;
    let size_lo = 10;

    let learning_rate = 0.005;

    let mut li = na::DMatrix::from_element(size_l1, 1, 0.);
    let mut lo = na::DMatrix::from_element(size_lo, 1, 0.);

    let mut nna = nn::NN::new(vec![           size_li,           size_l1,           size_l2,              size_lo],
                              vec!["relu".to_string(),"relu".to_string(),"relu".to_string(),"softmax".to_string()], 
                              learning_rate, 
                              false);

    nna.enable_dp(true, 0.01, 1.0);

    nna.train(x_train, y_train, 1, 1, 1, 1);
    nna.test(x_test, y_test);

        */


    Ok(())
}   

