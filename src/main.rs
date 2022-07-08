
use postgres::{Client, Error, NoTls};

extern crate nalgebra as na;

mod fact;
mod dp;
mod lr;
mod nnlayer;
mod nn;


type VecVec64 = Vec<Vec<Option<f64>>>;


use rand::thread_rng;
use rand::seq::SliceRandom;






fn main() -> Result<(), Error> {


    /*
    let _epsilon = 3.0;
    let _noise_scale = 0.01;
    let _data_norm = 7.89;

    let epochs = 1000;
    let batch = 50;
    let nfeat = 4;
    let nclass = 3;

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

    let mut size_cv = 0.00;
    let mut size_test = 0.00;
    let mut size_train = 1.0 - size_cv - size_test;

    let mut i : i64 = 0;
    let split_train : i64 = (sn.len() as f64 * size_train) as i64;
    let split_cv : i64 = (sn.len() as f64 * (size_train + size_cv)) as i64;
    for n in sn {
        if i < split_train {
            x_train.push(n[0..nfeat].to_vec());
            y_train.push(n[nfeat].clone());
        } else if i < split_cv {
            x_cv.push(n[0..nfeat].to_vec());
            y_cv.push(n[nfeat].clone());
        } else {
            x_test.push(n[0..nfeat].to_vec());
            y_test.push(n[nfeat].clone());
        }
        i = i + 1;
    }



    let mut lr = lr::LogisticRegression::new(epochs, batch, nfeat, nclass, 0.01, 0.001, false);

    lr.fit(x_train.clone(), y_train.clone(), epochs as usize, batch as usize);
    lr.test(x_train.clone(), y_train.clone());
    let loss1 = lr.get_loss();

    lr.reset();

    lr.enable_dp(true, _epsilon, _noise_scale, _data_norm);
    lr.fit(x_train.clone(), y_train.clone(), epochs_dp as usize, batch_dp as usize);
    lr.test(x_train.clone(), y_train.clone());
    let loss2 = lr.get_loss();
    */

    /*
    let mut lr = lr::LogisticRegression::new(epochs, batch, nfeat, nclass, 0.1, 0.01, false);
    lr.enable_dp(true, 1.0, 0.01, 1.0);

    lr.fit(x_train.clone(), y_train.clone());
    lr.test(x_train.clone(), y_train.clone());
    */




    




    //let _epsilon = 3.0;
    //let _noise_scale = 0.01;
    //let _data_norm = 7.89;

    let _epsilon = 1.0;
    let _noise_scale = 1.0;
    let _data_norm = 1000.0;

    let epochs = 5;
    let batch = 1000;
    let epochs_dp = 5;
    let batch_dp = 1000;
    let nfeat = 20;
    let nclass = 10;

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

    let mut _size_train = 0.60;
    let mut _size_cv = 0.20;
    let mut _size_test = 0.20;

    let mut i : i64 = 0;
    let split_train : i64 = (sn.len() as f64 * _size_train) as i64;
    let split_cv : i64 = (sn.len() as f64 * (_size_train + _size_cv)) as i64;
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

    let mut lr = lr::LogisticRegression::new(epochs, batch, nfeat, nclass, 0.01, 0.001, false);

    lr.fit(x_train.clone(), y_train.clone(), epochs as usize, batch as usize);
    lr.test(x_train.clone(), y_train.clone());
    let loss1 = lr.get_loss();

    lr.reset();

    lr.enable_dp(true, _epsilon, _noise_scale, _data_norm);
    lr.fit(x_train.clone(), y_train.clone(), epochs_dp as usize, batch_dp as usize);
    lr.test(x_train.clone(), y_train.clone());
    let loss2 = lr.get_loss();

    



    /*
    let size_li = x_train[0].len();
    let size_l1 = 50;
    let size_l2 = 25;
    let size_lo = 10;

    let learning_rate = 0.005;

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

