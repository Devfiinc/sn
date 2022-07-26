use postgres::{Client, Error, NoTls};

use rand::thread_rng;
use rand::seq::SliceRandom;

extern crate nalgebra as na;

mod fact;
mod dp;
mod lr;
mod nnlayer;
mod nn;




fn _print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}



fn _query_col_string(query : &str, idx : usize, data : &mut Vec<Vec<String>>) -> Result<(), Error> {
    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();

    for row in conn.query(query, &[])? {
        let mut v : Vec<String> = Vec::new();
        v.push(row.get(idx));
        data.push(v);
    }
    Ok(())
}




fn query_vec(query : &str, data : &mut Vec<Vec<f64>>) -> Result<(), Error> {
    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();

    for row in conn.query(query, &[])? {
        let mut v : Vec<f64> = Vec::new();
        for i in 0..row.len() {
            v.push(row.get(i));
        }
        data.push(v);
    }
    Ok(())
}




fn _query_vec_i64_f64(query : &str, data : &mut Vec<Vec<f64>>) -> Result<(), Error> {
    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();

    for row in conn.query(query, &[])? {
        let mut v : Vec<i32> = Vec::new();
        for i in 0..row.len() {
            v.push(row.get(i));
        }
        let mut v1 : Vec<f64> = Vec::new();
        for i in 0..v.len(){
            v1.push(v[i] as f64 / 255.0);
        }
        data.push(v1);
    }
    Ok(())
}




fn query_images(query : &str, data : &mut Vec<Vec<f64>>) -> Result<(), Error> {
    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();

    for row in conn.query(query, &[])? {
        let mut v : Vec<i32> = Vec::new();
        v = row.get(0);
        // for i in 0..row.len() {
        //     v.push(row.get(i));
        // }
        let mut v1 : Vec<f64> = Vec::new();
        for i in 0..v.len(){
            v1.push(v[i] as f64 / 255.0);
        }
        data.push(v1);
    }
    Ok(())
}




fn _query_images_2d(query : &str, data : &mut Vec<Vec<Vec<f64>>>) -> Result<(), Error> {
    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();

    for row in conn.query(query, &[])? {
        let mut v : Vec<Vec<i32>> = Vec::new();
        v = row.get(0);
        // for i in 0..row.len() {
        //     v.push(row.get(i));
        // }
        let mut v0 : Vec<Vec<f64>> = Vec::new();
        for i in 0..v.len(){
            let mut v1 : Vec<f64> = Vec::new();
            for j in 0..v[i].len(){
                v1.push(v[i][j] as f64 / 255.0);
            }
            v0.push(v1);
        }
        data.push(v0);
    }
    Ok(())
}




fn query_labels(query : &str, data : &mut Vec<Vec<f64>>) -> Result<(), Error> {
    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();

    for row in conn.query(query, &[])? {
        let v : i32 = row.get(0);
        let mut v1 : Vec<f64> = Vec::new();
        v1.push(v as f64);
        data.push(v1);
    }
    Ok(())
}






fn query_vec_range(query : &str, i1 : usize, i2 : usize, data : &mut Vec<Vec<f64>>) -> Result<(), Error> {
    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();

    let mut i2i = i2;

    for row in conn.query(query, &[])? {
        if i2i > row.len() {
            i2i = row.len();
        }
        let mut v : Vec<f64> = Vec::new();
        for i in i1..i2i {
            v.push(row.get(i));
        }
        data.push(v);
    }
    Ok(())
}



fn split_dataset(data    : Vec<Vec<f64>>, yid     : usize,
                 x_train : &mut Vec<Vec<f64>>, y_train : &mut Vec<f64>,
                 x_cv    : &mut Vec<Vec<f64>>, y_cv    : &mut Vec<f64>,
                 x_test  : &mut Vec<Vec<f64>>, y_test  : &mut Vec<f64>) {

    let mut _size_train = 0.60;
    let mut _size_cv = 0.20;
    let mut _size_test = 0.20;

    let mut i : i64 = 0;
    let split_train : i64 = (data.len() as f64 * _size_train) as i64;
    let split_cv : i64 = (data.len() as f64 * (_size_train + _size_cv)) as i64;
    for n in data {
        if i < split_train {
            x_train.push(n[0..yid].to_vec());
            y_train.push(n[yid].clone());
        } else if i < split_cv {
            x_cv.push(n[0..yid].to_vec());
            y_cv.push(n[yid].clone());
        } else {
            x_test.push(n[0..yid].to_vec());
            y_test.push(n[yid].clone());
        }
        i = i + 1;
    }
}



fn split_dataset_xy(dx      : Vec<Vec<f64>>,      dy      : Vec<Vec<f64>>,
                    x_train : &mut Vec<Vec<f64>>, y_train : &mut Vec<f64>,
                    x_cv    : &mut Vec<Vec<f64>>, y_cv    : &mut Vec<f64>,
                    x_test  : &mut Vec<Vec<f64>>, y_test  : &mut Vec<f64>) {

    let mut _size_train = 0.70;
    let mut _size_cv = 0.15;
    let mut _size_test = 0.15;
                    
    let mut i : i64 = 0;
    let split_train : i64 = (dx.len() as f64 * _size_train) as i64;
    let split_cv : i64 = (dx.len() as f64 * (_size_train + _size_cv)) as i64;
    for n in 0..dx.len() {
        if i < split_train {
            x_train.push(dx[n].clone());
            y_train.push(dy[n][0].clone());
        } else if i < split_cv {
            x_cv.push(dx[n].clone());
            y_cv.push(dy[n][0].clone());
        } else {
            x_test.push(dx[n].clone());
            y_test.push(dy[n][0].clone());
        }
        i = i + 1;
    }
}



fn split_dataset_xy_mdim(dx      : Vec<Vec<f64>>,      dy      : Vec<Vec<f64>>,
                         x_train : &mut Vec<Vec<f64>>, y_train : &mut Vec<Vec<f64>>,
                         x_cv    : &mut Vec<Vec<f64>>, y_cv    : &mut Vec<Vec<f64>>,
                         x_test  : &mut Vec<Vec<f64>>, y_test  : &mut Vec<Vec<f64>>) {

    let mut _size_train = 0.70;
    let mut _size_cv = 0.15;
    let mut _size_test = 0.15;
                    
    let mut i : i64 = 0;
    let split_train : i64 = (dx.len() as f64 * _size_train) as i64;
    let split_cv : i64 = (dx.len() as f64 * (_size_train + _size_cv)) as i64;
    for n in 0..dx.len() {
        if i < split_train {
            x_train.push(dx[n].clone());
            y_train.push(dy[n].clone());
        } else if i < split_cv {
            x_cv.push(dx[n].clone());
            y_cv.push(dy[n].clone());
        } else {
            x_test.push(dx[n].clone());
            y_test.push(dy[n].clone());
        }
        i = i + 1;
    }
}



fn _split_dataset_xy_mdim_2d(dx      : Vec<Vec<Vec<f64>>>, dy      : Vec<Vec<Vec<f64>>>,
                            x_train : &mut Vec<Vec<Vec<f64>>>, y_train : &mut Vec<Vec<Vec<f64>>>,
                            x_cv    : &mut Vec<Vec<Vec<f64>>>, y_cv    : &mut Vec<Vec<Vec<f64>>>,
                            x_test  : &mut Vec<Vec<Vec<f64>>>, y_test  : &mut Vec<Vec<Vec<f64>>>) {

    let mut _size_train = 0.70;
    let mut _size_cv = 0.15;
    let mut _size_test = 0.15;
                    
    let mut i : i64 = 0;
    let split_train : i64 = (dx.len() as f64 * _size_train) as i64;
    let split_cv : i64 = (dx.len() as f64 * (_size_train + _size_cv)) as i64;
    for n in 0..dx.len() {
        if i < split_train {
            x_train.push(dx[n].clone());
            y_train.push(dy[n].clone());
        } else if i < split_cv {
            x_cv.push(dx[n].clone());
            y_cv.push(dy[n].clone());
        } else {
            x_test.push(dx[n].clone());
            y_test.push(dy[n].clone());
        }
        i = i + 1;
    }
}






fn logistic_regression(db : &str, _class : bool) {

    let mut query = String::new();
    query.push_str("SELECT * from ");
    query.push_str(db);


    let mut data: Vec<Vec<f64>> = vec![];

    if db == "nki" {

        let res = query_vec_range(&query, 2, 20, &mut data);
        println!("{:?}", res);

    } else if db == "sn" {

        let res = query_vec(&query, &mut data);
        println!("{:?}", res);

    }

    println!("{}", db);
    println!("{}", data[0][0]);



    let _epsilon = 1.0;
    let _noise_scale = 1.0;
    let _data_norm = 1000.0;

    let epochs = 5;
    let batch = 1000;
    let epochs_dp = 5;
    let batch_dp = 1000;
    let nfeat = 20;
    let nclass = 10;

    
    // Shuffle input
    data.shuffle(&mut thread_rng());

    // Split dataset into train, cross validation and test
    let mut x_train : Vec<Vec<f64>> = vec![];
    let mut y_train : Vec<f64> = vec![];
    let mut x_cv : Vec<Vec<f64>> = vec![];
    let mut y_cv : Vec<f64> = vec![];
    let mut x_test : Vec<Vec<f64>> = vec![];
    let mut y_test : Vec<f64> = vec![];

    split_dataset(data, 20, &mut x_train, &mut y_train, &mut x_cv, &mut y_cv, &mut x_test, &mut y_test);

    let mut lr = lr::LogisticRegression::new(epochs, batch, nfeat, nclass, 0.01, 0.001, true, false);

    lr.fit(x_train.clone(), y_train.clone(), epochs as usize, batch as usize);
    lr.test(x_train.clone(), y_train.clone());
    let loss1 = lr.get_loss();

    lr.reset();

    lr.enable_dp(true, _epsilon, _noise_scale, _data_norm);
    lr.fit(x_train.clone(), y_train.clone(), epochs_dp as usize, batch_dp as usize);
    lr.test(x_train.clone(), y_train.clone());
    let loss2 = lr.get_loss();

    println!("{}", loss1);
    println!("{}", loss2);

}



fn neural_network(db : &str, topology : Vec<usize>) {

    let mut query = String::new();
    query.push_str("SELECT * from ");
    query.push_str(db);

    let mut data: Vec<Vec<f64>> = vec![];
    let _qvec = query_vec(&query, &mut data);


    let _epsilon = 1.0;
    let _noise_scale = 1.0;
    let _data_norm = 1000.0;

    let _epochs = 5;
    let _batch = 1000;
    let _epochs_dp = 5;
    let _batch_dp = 1000;
    let _nfeat = 20;
    let _nclass = 10;

    // Shuffle input
    data.shuffle(&mut thread_rng());

    // Split dataset into train, cross validation and test
    let mut x_train : Vec<Vec<f64>> = vec![];
    let mut y_train : Vec<f64> = vec![];
    let mut x_cv : Vec<Vec<f64>> = vec![];
    let mut y_cv : Vec<f64> = vec![];
    let mut x_test : Vec<Vec<f64>> = vec![];
    let mut y_test : Vec<f64> = vec![];
    split_dataset(data, 20, &mut x_train, &mut y_train, &mut x_cv, &mut y_cv, &mut x_test, &mut y_test);

    let mut _topology : Vec<usize> = vec![x_train[0].len()];
    for i in topology.clone() {
        _topology.push(i);
    }
    
    let mut _facts : Vec<String> = vec![];
    for _ in 0..topology.len()-1 {
        _facts.push("relu".to_string());
    }
    _facts.push("softmax".to_string());


    let learning_rate = 0.005;

    let mut nna = nn::NN::new(vec!["dense".to_string()],
                              _topology,
                              _facts, 
                              learning_rate, 
                              false);

    nna.enable_dp(true, _epsilon, _noise_scale, _data_norm);

    nna.train(x_train, y_train, x_cv, y_cv, 1, 1, 1, 1);
    nna.test(x_test, y_test);
}





fn neural_network_mnist(topology : Vec<usize>) {

    let query_x = String::from("SELECT image FROM mnist LIMIT 10000");
    let query_y = String::from("SELECT label FROM mnist LIMIT 10000");

    let mut data_x: Vec<Vec<f64>> = vec![];
    let _qimg = query_images(&query_x, &mut data_x);

    let mut data_y: Vec<Vec<f64>> = vec![];
    let _qlbl = query_labels(&query_y, &mut data_y);

    let _epsilon = 1.0;
    let _noise_scale = 1.0;
    let _data_norm = 1000.0;

    let _epochs = 5;
    let _batch = 1000;
    let _epochs_dp = 5;
    let _batch_dp = 1000;
    let _nfeat = 20;
    let _nclass = 10;

    // Split dataset into train, cross validation and test
    let mut x_train : Vec<Vec<f64>> = vec![];
    let mut y_train : Vec<f64> = vec![];
    let mut x_cv : Vec<Vec<f64>> = vec![];
    let mut y_cv : Vec<f64> = vec![];
    let mut x_test : Vec<Vec<f64>> = vec![];
    let mut y_test : Vec<f64> = vec![];
    split_dataset_xy(data_x, data_y, &mut x_train, &mut y_train, &mut x_cv, &mut y_cv, &mut x_test, &mut y_test);


    let mut _topology : Vec<usize> = vec![x_train[0].len()];
    for i in topology.clone() {
        _topology.push(i);
    }
    
    let mut _facts : Vec<String> = vec![];
    for _ in 0..topology.len()-1 {
        _facts.push("sigmoid".to_string());
    }
    _facts.push("softmax".to_string());

    let learning_rate = 0.005;

    let mut nna = nn::NN::new(vec!["conv2d".to_string()],
                              _topology,
                              _facts, 
                              learning_rate, 
                              false);

    nna.enable_dp(true, _epsilon, _noise_scale, _data_norm);

    nna.train_1(x_train.clone(), y_train.clone(), x_cv.clone(), y_cv.clone(), 10, 1, 1, 1, (28,28));
}


fn neural_network_nerves(topology : Vec<usize>) {

    let query_x = String::from("SELECT imx from nerves WHERE imx IS NOT NULL AND imy IS NOT NULL LIMIT 1000");
    let query_y = String::from("SELECT imy from nerves WHERE imy IS NOT NULL AND imx IS NOT NULL LIMIT 1000");

    let mut data_x: Vec<Vec<f64>> = vec![];
    let _qimgs = query_images(&query_x, &mut data_x);

    let mut data_y: Vec<Vec<f64>> = vec![];
    let _qlbl = query_images(&query_y, &mut data_y);

    let _epsilon = 1.0;
    let _noise_scale = 1.0;
    let _data_norm = 1000.0;

    let _epochs = 5;
    let _batch = 1000;
    let _epochs_dp = 5;
    let _batch_dp = 1000;
    let _nfeat = 20;
    let _nclass = 10;

    // Split dataset into train, cross validation and test
    let mut x_train : Vec<Vec<f64>> = vec![];
    let mut y_train : Vec<Vec<f64>> = vec![];
    let mut x_cv    : Vec<Vec<f64>> = vec![];
    let mut y_cv    : Vec<Vec<f64>> = vec![];
    let mut x_test  : Vec<Vec<f64>> = vec![];
    let mut y_test  : Vec<Vec<f64>> = vec![];

    split_dataset_xy_mdim(data_x, data_y, &mut x_train, &mut y_train, &mut x_cv, &mut y_cv, &mut x_test, &mut y_test);

    let mut _topology : Vec<usize> = vec![x_train[0].len()];
    for i in topology.clone() {
        _topology.push(i);
    }
    
    let mut _facts : Vec<String> = vec![];
    for _ in 0..topology.len()-1 {
        _facts.push("sigmoid".to_string());
    }
    _facts.push("softmax".to_string());

    let learning_rate = 0.05;

    let mut nna = nn::NN::new(vec!["conv2d".to_string()],
                              _topology,
                              _facts, 
                              learning_rate, 
                              false);

    /*nna.enable_dp(true, 0.01, 1.0);*/

    nna.train_mdim_1(x_train.clone(), y_train.clone(), x_train.clone(), y_train.clone(), 5, 1, 1, 1, (96, 96));
}







fn main() -> Result<(), Error> {

    let opt : usize = 5;

    if opt == 1{    /*Done*/

        println!("Logistic Regression - Classifier - Tabular Data - SmartNoise/random");
        logistic_regression("sn", true);

    } else if opt == 2 {    /*Done*/

        println!("Neural Network Classifier - Tabular Data - SmartNoise/random");
        neural_network("sn", vec![25,25,10]);

    } else if opt == 3 {

        println!("Logistic Regression - Continuous - Tabular Data - Kaggle/nki");
        logistic_regression("nki", false);

    } else if opt == 4 {

        println!("Neural Network Continuous - Tabular Data - Kaggle/nki");
        neural_network("nki", vec![25,25,10]);

    } else if opt == 5 {
        println!("Neural Network - Classifier - Image data - MNist");
        neural_network_mnist(vec![128,64,10]);

    } else if opt == 6 {

        println!("Neural Network - Classifier - Image data - Kaggle nerve dataset");
        neural_network_nerves(vec![128,64,10]);

    }

    Ok(())
}   

