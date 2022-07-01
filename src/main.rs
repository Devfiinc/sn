//use opendp::core::*;
use postgres::{Client, Error, NoTls};
//use opendp::*;
use ndarray::*;

extern crate nalgebra as na;
use rand::Rng;
use na::{DMatrix, Hessenberg, Matrix4};

pub mod nn;


type VecVec64 = Vec<Vec<Option<f64>>>;






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
    
    //println!("{}", sn[0][0].unwrap());



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
            y_train_1.push(n[20]);
        } else {
            x_test_1.push(n[0..20].to_vec());
            y_test_1.push(n[20]);
        }
        i = i + 1;
    }





    


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



    let size_li = x_train.dim().1;
    let size_l1 = 30;
    let size_l2 = 30;
    let size_lo = 10;
    let learning_rate = 0.001;

    let mut wi1 = na::DMatrix::from_fn(size_li + 1, size_l1, |r,c| {rand::random::<f64>() - 0.5});
    let mut w12 = na::DMatrix::from_fn(size_l1 + 1, size_l2, |r,c| {rand::random::<f64>() - 0.5});
    let mut w2o = na::DMatrix::from_fn(size_l2 + 1, size_lo, |r,c| {rand::random::<f64>() - 0.5});

    let mut li = na::DMatrix::from_element(size_l1, 1, 0.);
    let mut l1 = na::DMatrix::from_element(size_l1, 1, 0.);
    let mut l2 = na::DMatrix::from_element(size_l2, 1, 0.);
    let mut lo = na::DMatrix::from_element(size_lo, 1, 0.);
    
    let mut a1 = na::DMatrix::from_element(size_l1 + 1, 1, 0.);
    let mut a2 = na::DMatrix::from_element(size_l2 + 1, 1, 0.);
    let mut ao = na::DMatrix::from_element(size_lo, 1, 0.);

    let mut bi = na::DMatrix::from_element(size_li + 1, 1, 0.);
    let mut b1 = na::DMatrix::from_element(size_l1 + 1, 1, 0.);
    let mut b2 = na::DMatrix::from_element(size_l2 + 1, 1, 0.);

    bi[(0,0)] = 1.;
    b1[(0,0)] = 1.;
    b2[(0,0)] = 1.;

    let mut delta1 = na::DMatrix::from_element(size_l1 + 1, 1, 0.);
    let mut delta2 = na::DMatrix::from_element(size_l2 + 1, 1, 0.);
    let mut deltao = na::DMatrix::from_element(size_lo, 1, 0.);

    let mut cli = nn::NN::new(vec![size_li, size_l1, size_l2, size_lo], 0.01, false);


    let mut idx : usize = 0;
    for x in x_train_1 {
        //let mut li = na::DMatrix::<f64>::from_vec(size_li, 1, x.iter().map(|x| x.unwrap()).collect());
        //let zi = li + &bi;
        li = na::DMatrix::<f64>::from_vec(size_li, 1, x.iter().map(|x| x.unwrap()).collect());
        li = li.insert_row(0, 1.0);
        //let zi = li;

        println!("l1");
        l1 = li.transpose() * &wi1;
        l1 = l1.insert_column(0, 1.0);
        a1 = l1.map(|x| nn::NN::sigmoid(x));

        println!("l2");
        l2 = l1 * &w12;
        l2 = l2.insert_column(0, 1.0);
        a2 = l2.map(|x| nn::NN::sigmoid(x));

        println!("lo");
        lo = l2 * &w2o;
        ao = lo.map(|x| nn::NN::sigmoid(x));
        
        println!("ly");
        let y : i64 = y_train_1[idx].unwrap() as i64;
        let mut ly = na::DMatrix::from_element(size_lo, 1, 0.);
        ly[(y as usize,0)] = 1.;




        deltao = lo - ly.transpose();

        println!("deltao {} {}", deltao.shape().0, deltao.shape().1);
        println!("w2o    {} {}", w2o.shape().0, w2o.shape().1);

        delta2 = deltao * w2o.transpose();

        println!("delta2 {} {}", delta2.shape().0, delta2.shape().1);
        println!("w12    {} {}", w12.shape().0, w12.shape().1);

        //delta1 = delta2 * w12.transpose();




        /*
        
        println!("deltao");
        deltao = lo - ly.transpose();
        println!("{} {}", deltao[(0,0)], deltao[(0,1)]);
        
        println!("deltao");
        delta2 = deltao * w2o.transpose(); // * a2.map(|x| nn::NN::sigmoid_derivative(x));
        delta2 = delta2.component_mul(&a2.map(|x| nn::NN::sigmoid_derivative(x)));
        println!("{} {}", delta2[(0,0)], delta2[(0,1)]);
        
        println!("delta2 {} {}", delta2.shape().0, delta2.shape().1);
        println!("w12 {} {}", w12.shape().0, w12.shape().1);
        println!("delta1 {} {}", delta1.shape().0, delta1.shape().1);
        println!("a1 {} {}", a1.shape().0, a1.shape().1);
        delta1 = delta2 * w12;
        delta1 = delta1.component_mul(&a1.map(|x| nn::NN::sigmoid_derivative(x)));
        println!("{} {}", delta1[(0,0)], delta1[(0,1)]);
        
        break;
        
        println!("w");
        //w2o = w2o.clone() - &(deltao * &a2.transpose() * learning_rate);
        //w12 = w12.clone() - &(delta2 * &a1.transpose() * learning_rate);
        //wi1 = wi1.clone() - &(delta1 * &li.transpose() * learning_rate);

        // * &lo.map(|x| nn::NN::sigmoid_derivative(x))

    

        break;
        
        */

        println!("{}", idx);
        idx += 1;
    }



    /*







    let mut cli = lr::LR::new("Opt");
    let cliid = cli.get_id();
    println!("{}",cliid);

    let size_li = 20;
    let size_l1 = 30;
    let size_l2 = 30;
    let size_lo = 10;



    println!("Dims x_train = {:?} and y_train = {}", x_train.shape()[0], y_train.ndim());

    // Input shape 20 -> each row in x_train & x_test
    // x_train.iter()
    // Shape = x_train.dim().1 = 20
    // let mut lo = Array::from_elem((1, size_in), 0.);

    // Layer 1
    let mut w1 = Array::from_elem((size_li, size_l1), 0.);
    //let mut l1 = Array::from_elem((1, size_l1), 0.);

    // Layer 2
    let mut w2 = Array::from_elem((size_l1, size_l2), 0.);
    //let mut l2 = Array::from_elem((1, size_l2), 0.);

    
    // Output shape 10 (num of classes) -> match with y_train & y_test
    let mut wo = Array::from_elem((size_l2, size_lo), 0.);
    //let mut lo = Array::from_elem((1, size_out), 0.);




    let ilen = x_train.dim().0 as i64;
    let jlen = x_train.dim().1 as i64;

    println!("Iterator {} x {}", ilen, jlen);





    // Train model on data
    for row in x_train.genrows() {
        
        let mut l0 = Array::from_elem((1, 20), 0.);
        for j in 0..jlen-1 {
            l0[[0, j as usize]] = row[[j as usize]];
            //println!("{:?}", l0[[j as usize]]);
        }

        //println!("{}", type(l0));
        //println!("{} {} {} {} {} {}", l0.dim().0, l0.dim().1, w1.dim().0, w1.dim().1, l1.dim().0, l1.dim().1);

        // Forward
        let mut l1 = l0 * w1.clone();
        //l1 = fact(l1);
        let mut l2 = l1 * w2.clone();
        //l2 = fact(l2);
        let mut lo = l2 * wo.clone();
        //lo = fact(lo);

        // Error

        // Backward

    }


    // Test model on data
    for n in x_test {

        // Forward

        // Error


    }

*/
    Ok(())

}   

