//use opendp::core::*;
use postgres::{Client, Error, NoTls};
//use opendp::*;
use ndarray::*;

pub mod lr;

type Vec64_20 = Vec<(Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>)>;

type VecVec64 = Vec<Vec<Option<f64>>>;

type Vec64_21 = Vec<(Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>)>;


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
            x_train_1.push(n[0..19].to_vec());
            y_train_1.push(n[20]);
        } else {
            x_test_1.push(n[0..19].to_vec());
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

    
    for i in 0..ilen-1 {
        for j in 0..jlen-1 {
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
        for j in 0..jlen-1 {
            x_test[[i as usize,j as usize]] = x_test_1[i as usize][j as usize].unwrap();
        }
        y_test[[i as usize,0]] = y_test_1[i as usize].unwrap();
    }




    let mut cli = lr::LR::new("Opt");
    let cliid = cli.get_id();
    println!("{}",cliid);

    let size_l0 = 20;
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









    
/*
    use opendp::meas::make_base_gaussian;

    let epsilon = 3.0;
    let data_norm = 7.89;

    let meas1 = opendp::meas::make_base_gaussian(epsilon, X_train);
    let meas2 = opendp::meas::make_base_gaussian(epsilon, X_test);
*/  

/*
    // Privatelly transform the data
    use opendp::trans::*;
    use opendp::comb::*;
    use opendp::meas::*;


    
    let trans0 = opendp::trans::make
    let trans1 = opendp::trans::make_split_dataframe(separator: Option<&str>, col_names: Vec<K>)?;
    let trans2 = opendp::trans::make_split_records(separator: Option<&str>)?;

    let cast = make_cast_default::<f64, f64>()?;
    let load_numbers = make_chain_tt(&cast, &trans2, None)?;


    let clamp = make_clamp(bounds)?;
    let bounded_sum = make_bounded_sum(bounds)?;
    let laplace = make_base_laplace(sigma)?;
    let intermediate = make_chain_tt(&bounded_sum, &clamp, None)?;
    let noisy_sum = make_chain_mt(&laplace, &intermediate, None)?;

    // Get the data privatelly

    //let privac = opendp::trans::
    //let clf = new LogisticsRegression();

*/




    /*
    for row in conn.query("SELECT id, username, password, email FROM users", &[])? {
        let id: i32 = row.get(0);
        let username: &str = row.get(1);
        let password: &str = row.get(2);
        let email: &str = row.get(3);
        println!(
            "found app user: {}) {} | {} | {}",
            id, username, password, email
        );
    }
    */


    Ok(())

}   



/*
fn main() -> Result<(), Error> {
    let client = Client::connect("postgres://postgres:postgres@localhost:5432", NoTls)?;
    let mut conn = client.get_txn()?;
    let mut query = conn.query("SELECT * FROM users", &[])?;
    let mut users: Vec<User> = Vec::new();
    while let Some(row) = query.next()? {
        let user: User = row.get(0);
        users.push(user);
    }
    println!("{:?}", users);
    Ok(())
}



fn main() {
    println!("Hello, world!");
}


*/


