//use opendp::core::*;
use postgres::{Client, Error, NoTls};
//use opendp::*;
use ndarray::*;

extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

pub mod nn;


type VecVec64 = Vec<Vec<Option<f64>>>;


use rand::thread_rng;
use rand::seq::SliceRandom;




fn main() -> Result<(), Error> {

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
    
    
    // Randomize input
    sn.shuffle(&mut thread_rng());


    // Split dataset into train and test
    let mut x_train_1 : VecVec64 = vec![];
    let mut y_train_1 : Vec<Option<f64>> = vec![];
    let mut x_test_1 : VecVec64 = vec![];
    let mut y_test_1 : Vec<Option<f64>> = vec![];

    let mut i : i64 = 0;
    let split : i64 = (sn.len() as f64 * 0.9) as i64;
    for n in sn {
        if i < split {
            x_train_1.push(n[0..20].to_vec());
            y_train_1.push(n[20].clone());
        } else {
            x_test_1.push(n[0..20].to_vec());
            y_test_1.push(n[20].clone());
        }
        i = i + 1;
    }

    for i in 0..100 {
        println!("{} {} {}", x_train_1[i][0].unwrap(), x_train_1[i][1].unwrap(), y_train_1[i].unwrap());

    }

/*
    let mut cnt : Vec<i64> = vec![0; 10];

    for n in y_test_1 {
        let a = n.unwrap() as i64 as usize;
        cnt[a] = cnt[a] + 1;
    }

    for i in 0..10 {
        println!("{}", cnt[i]);
    }
    */

    /*
    for i in 0..y_train_1.len() {
        cnt[y_train_1[i].unwrap() as usize] = cnt[y_train_1[i].unwrap() as usize] + 1;
    }
    */


    
/*

    let mut x_train = Array::from_elem((x_train_1.len(), 20), 0.);
    let mut y_train = Array::from_elem((y_train_1.len(), 1), 0.);
    println!("Dimensions train = {} x {}", x_train.dim().0, x_train.dim().1);

    let ilen = x_train.dim().0 as i64;
    let jlen = x_train.dim().1 as i64;

    println!("Iterator {} x {}", ilen, jlen);

    
    for i in 0..ilen {
        for j in 0..jlen {
            x_train[[i as usize, j as usize]] = x_train_1[i as usize][j as usize].unwrap();
        }
        y_train[[i as usize,0]] = y_train_1[i as usize].unwrap();
    }
    

    let mut x_test = Array::from_elem((x_test_1.len(), 20), 0.);
    let mut y_test = Array::from_elem((y_test_1.len(), 1), 0.);
    println!("Dimensions test = {} x {}", x_test.dim().0, x_test.dim().1);

    let ilen = x_test.dim().0 as i64;
    let jlen = x_test.dim().1 as i64;

    for i in 0..ilen-1 {
        for j in 0..jlen {
            x_test[[i as usize,j as usize]] = x_test_1[i as usize][j as usize].unwrap();
        }
        y_test[[i as usize,0]] = y_test_1[i as usize].unwrap();
    }


*/




    

    let size_li = x_train_1[0].len();
    let size_l1 = 30;
    let size_l2 = 30;
    let size_lo = 10;
    let learning_rate = 0.001;

    let mut wi1 = na::DMatrix::from_fn(size_li, size_l1, |r,c| {rand::random::<f64>() - 0.5});
    let mut w12 = na::DMatrix::from_fn(size_l1, size_l2, |r,c| {rand::random::<f64>() - 0.5});
    let mut w2o = na::DMatrix::from_fn(size_l2, size_lo, |r,c| {rand::random::<f64>() - 0.5});

    let mut li = na::DMatrix::from_element(size_l1, 1, 0.);
    let mut l1 = na::DMatrix::from_element(size_l1, 1, 0.);
    let mut l2 = na::DMatrix::from_element(size_l2, 1, 0.);
    let mut lo = na::DMatrix::from_element(size_lo, 1, 0.);
    
    let mut a1 = na::DMatrix::from_element(size_l1, 1, 0.);
    let mut a2 = na::DMatrix::from_element(size_l2, 1, 0.);
    let mut ao = na::DMatrix::from_element(size_lo, 1, 0.);

    let mut bi = na::DMatrix::from_element(size_li, 1, 0.);
    let mut b1 = na::DMatrix::from_element(size_l1, 1, 0.);
    let mut b2 = na::DMatrix::from_element(size_l2, 1, 0.);

    bi[(0,0)] = 1.;
    b1[(0,0)] = 1.;
    b2[(0,0)] = 1.;

    let mut delta1 = na::DMatrix::from_element(size_l1, 1, 0.);
    let mut delta2 = na::DMatrix::from_element(size_l2, 1, 0.);
    let mut deltao = na::DMatrix::from_element(size_lo, 1, 0.);

    let mut cli = nn::NN::new(vec![size_li, size_l1, size_l2, size_lo], 0.01, false);


    let mut idx : usize = 0;
    for x in x_train_1 {
        li = na::DMatrix::<f64>::from_vec(size_li, 1, x.iter().map(|x| x.unwrap()).collect());
        println!("li {}", li[(0,0)]);
        println!("wi1 {}", wi1[(0,0)]);

        l1 = &li.transpose() * &wi1;
        a1 = l1.map(|x| nn::NN::sigmoid(x));

        println!("l1 {}", l1[(0,0)]);
        println!("a1 {}", a1[(0,0)]);

        l2 = l1 * &w12;
        a2 = l2.map(|x| nn::NN::sigmoid(x));

        lo = l2 * &w2o;
        ao = lo.map(|x| nn::NN::softmax(x));
        
        let y : i64 = y_train_1[idx].unwrap() as i64;
        let mut ly = na::DMatrix::from_element(size_lo, 1, 0.);
        ly[(y as usize,0)] = 1.;


        deltao = ao - ly.transpose();
        delta2 = &w2o * deltao.transpose();
        delta2 = delta2.component_mul(&a2.transpose().map(|x| nn::NN::softmax_derivative(x)));

        delta1 = &w12.transpose() * &delta2;
        delta1 = delta1.component_mul(&a1.transpose().map(|x| nn::NN::sigmoid_derivative(x)));

        w2o = w2o.clone() - &(deltao.transpose() * &a2 * learning_rate).transpose();
        w12 = w12.clone() - &(delta2 * &a1 * learning_rate).transpose();
        wi1 = wi1.clone() - &(delta1 * &li.transpose() * learning_rate).transpose();

        println!("{} {}", idx, y);
        idx += 1;

        if (idx == 1000) {
            break;
        }
    }









    let mut correct : i64 = 0;
    let mut total : i64 = 0;
    idx = 0;

    for x in x_test_1 {

        li = na::DMatrix::<f64>::from_vec(size_li, 1, x.iter().map(|x| x.unwrap()).collect());

        l1 = li.transpose() * &wi1;
        a1 = l1.map(|x| nn::NN::sigmoid(x));

        l2 = l1 * &w12;
        a2 = l2.map(|x| nn::NN::sigmoid(x));

        lo = l2 * &w2o;
        ao = lo.map(|x| nn::NN::softmax(x));

        let mut max_idx = 0;
        let mut max_val = 0.;
        for i in 0..ao.len(){
            print!("{} ", ao[(0,i)].clone());
            if ao[(0,i)] > max_val {
                max_val = ao[(0,i)].clone();
                max_idx = i;
            }
        }
        println!("");
        
        let y : i64 = y_test_1[idx].unwrap() as i64;
        if max_idx == y as usize {
            correct += 1;
        }

        total += 1;

        println!("{} - {}, {} / {} = {} %", y, max_idx, correct, total, (correct as f64 / total as f64) * 100.);
    }




    Ok(())

}   

