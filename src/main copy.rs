use opendp::core::*;
//use postgres::{Connection, TlsMode};
use postgres::{Client, Error, NoTls};





/*
fn create_table() {
    let mut client = Client::connect(
        "postgresql://dboperator:operatorpass123@localhost:5243/postgres",
        NoTls,
    );

    client.batch_execute(
        "
        CREATE TABLE IF NOT EXISTS users (
            id              SERIAL PRIMARY KEY,
            username        VARCHAR UNIQUE NOT NULL,
            password        VARCHAR NOT NULL,
            email           VARCHAR UNIQUE NOT NULL
            )
    ",
    );
}
*/

/*
fn insert_user(username: &str, password: &str, email: &str) {
    client.execute(
        "
        INSERT INTO users (username, password, email)
        VALUES ($1, $2, $3)
    ",
        &[&username, &password, &email],
    )?;
}
*/




fn main() {



    //let url = "postgresql://dboperator:operatorpass123@localhost:5243/postgres";
    let url = "postgresql://postgres:postgres@localhost:5432/postgres";
    let mut conn = Client::connect(url, NoTls).unwrap();


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


