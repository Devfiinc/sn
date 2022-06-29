use opendp::core::*;
use postgres::{Client, Error, NoTls};
use opendp::*;


type vec64_20 = Vec<(Option<f64>,
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

type vec64_21 = Vec<(Option<f64>,
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

/*
    let mut query = conn.query("
        CREATE TABLE IF NOT EXISTS users (
            id              SERIAL PRIMARY KEY,
            username        VARCHAR UNIQUE NOT NULL,
            password        VARCHAR NOT NULL,
            email           VARCHAR UNIQUE NOT NULL
            )
    ", &[]).unwrap();



    query = conn.query(
        "INSERT INTO users (username, password, email) VALUES ($1, $2, $3)",
        &[&"user1", &"mypass", &"user@test.com"],
    ).unwrap();


*/

    let mut sn : Vec<(Option<f64>,
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
                   Option<f64>)> = vec![];




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


        sn.push((var00,
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
                  var20));

        //println!(
        //    "row i : {}) {}",
        //    var00.unwrap(), var01.unwrap()
        //);
    }

    // Split dataset into train and test
    let mut X_train : vec64_20 = vec![];
    let mut X_test : vec64_20 = vec![];
    let mut y_train : Vec<Option<f64>> = vec![];
    let mut y_test : Vec<Option<f64>> = vec![];

    let mut i : i64 = 0;
    let split : i64 = (sn.len() as f64 * 0.8) as i64;
    for n in sn {
        if i < split {
            X_train.push((n.0, n.1, n.2, n.3, n.4, n.5, n.6, n.7, n.8, n.9, n.10, n.11, n.12, n.13, n.14, n.15, n.16, n.17, n.18, n.19));
            y_train.push(n.20);
        } else {
            X_test.push((n.0, n.1, n.2, n.3, n.4, n.5, n.6, n.7, n.8, n.9, n.10, n.11, n.12, n.13, n.14, n.15, n.16, n.17, n.18, n.19));
            y_test.push(n.20);
        }
        i = i + 1;
    }




    use opendp::meas::make_base_gaussian;

    let epsilon = 3.0;
    let data_norm = 7.89;

    let meas1 = opendp::meas::make_base_gaussian(epsilon, X_train);
    let meas2 = opendp::meas::make_base_gaussian(epsilon, X_test);
    

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


