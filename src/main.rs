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

        let var00 : Option<f64> = row.get(0);
        let var01 : Option<f64> = row.get(1);
        let var02 : Option<f64> = row.get(2);
        let var03 : Option<f64> = row.get(3);
        let var04 : Option<f64> = row.get(4);
        let var05 : Option<f64> = row.get(5);
        let var06 : Option<f64> = row.get(6);
        let var07 : Option<f64> = row.get(7);
        let var08 : Option<f64> = row.get(8);
        let var09 : Option<f64> = row.get(9);
        let var10 : Option<f64> = row.get(10);
        let var11 : Option<f64> = row.get(11);
        let var12 : Option<f64> = row.get(12);
        let var13 : Option<f64> = row.get(13);
        let var14 : Option<f64> = row.get(14);
        let var15 : Option<f64> = row.get(15);
        let var16 : Option<f64> = row.get(16);
        let var17 : Option<f64> = row.get(17);
        let var18 : Option<f64> = row.get(18);
        let var19 : Option<f64> = row.get(19);
        let var20 : Option<f64> = row.get(20);


        sn.push(vec![var00,
                     var01,
                     var02,
                     var03,
                     var04,
                     var05,
                     var06,
                     var07,
                     var08,
                     var09,
                     var10,
                     var11,
                     var12,
                     var13,
                     var14,
                     var15,
                     var16,
                     var17,
                     var18,
                     var19,
                     var20]);

        //println!(
        //    "row i : {}) {}",
        //    var00.unwrap(), var01.unwrap()
        //);
    }

    println!("{}", sn[0][0].unwrap());



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

    let w1 = na::DMatrix::from_fn(size_li, size_l1, |r,c| {rand::random::<f64>() - 0.5});
    let w2 = na::DMatrix::from_fn(size_l1, size_l2, |r,c| {rand::random::<f64>() - 0.5});
    let w3 = na::DMatrix::from_fn(size_l2, size_lo, |r,c| {rand::random::<f64>() - 0.5});

    let mut l1 = na::DMatrix::from_element(size_l1, 1, 0.);
    let mut l2 = na::DMatrix::from_element(size_l2, 1, 0.);
    let mut lo = na::DMatrix::from_element(size_lo, 1, 0.);


    let mut cli = nn::NN::new(vec![size_li, size_l1, size_l2, size_lo], 0.01, false);


    let mut idx : usize = 0;
    for x in x_train_1 {
        let li = na::DMatrix::<f64>::from_vec(size_li, 1, x.iter().map(|x| x.unwrap()).collect());

        l1 = li.transpose() * &w1;
        l1 = l1.map(|x| x.tanh());

        l2 = l1 * &w2;
        l2 = l2.map(|x| x.tanh());

        lo = l2 * &w3;
        lo = lo.map(|x| x.tanh());


        //let l1_sigmoid = l1.mapv(|x| 1. / (1. + (-x).exp()));

        //let l1 = li.dot(w1);
        //let dynamic_times_static: na::DVector<_> = dynamic_m * dynamic_m;




//        let l1 = Layer::new(size_l1, size_li, ActivationFunction::Sigmoid);
//
//        let l2 = Layer::new(size_l2, size_l1, ActivationFunction::Sigmoid);
//
//        let lo = Layer::new(size_lo, size_l2, ActivationFunction::Sigmoid);
//
//        let mut net = NeuralNetwork::new(li, l1, l2, lo);





        println!("{}", idx);
        idx += 1;
    }

    //let li = DMatrix::from_vec(size_li, size_l1, x_train.iter().map(|x| *x).collect());
    //let li = na::DMatrix::from_vec(size_li, size_l1, vec!(x_train[[0][0..size_li]].to_vec()));




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

