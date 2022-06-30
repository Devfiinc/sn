#! /bin/bash

#pg_lsclusters
sudo pg_ctlcluster 14 main start

#set user to group back with
sudo gpasswd -a postgres ssl-cert

# Fixed ownership and mode
sudo chown root:ssl-cert  /etc/ssl/private/ssl-cert-snakeoil.key
sudo chmod 740 /etc/ssl/private/ssl-cert-snakeoil.key


chmod -R 777 /usr/share/postgresql/14/extension/
chmod -R 777 /usr/lib/postgresql/14/lib/

# locate pg_hba.conf
# postgres -> trust
nano /etc/postgresql/14/main/pg_hba.conf



# now postgresql starts! (and install command doesn't fail anymore)
sudo service postgresql restart

psql postgres postgres
\password
postgres
postgres
\q


# below line is not needed in first instance, it's used for compiling already existing extensions.
su -c "cargo pgx run pg14" docker
su -c "cargo pgx run pg14 postgres" docker





# JUST TEST CODE FROM THIS POINT

#cargo new hello
#cargo init

#cargo run
#cargo build
#cargo build --release

#su -c "cargo pgx init --pg14=/usr/lib/postgresql/14/bin/pg_config" docker
#sudo su -c "cargo pgx run pg14" docker
#su -c "cargo pgx run pg14" docker

#cp -r ext1/ /home/docker
#chmod -R 777 ext1/
#su -c "cargo pgx run pg14" docker

#drop extension ext2; create extension ext2; SELECT hello_ext2(); SELECT to_lowercase('ASD'); \df

#CREATE TABLE IF NOT EXISTS emnist (
#	id INT PRIMARY KEY NOT NULL,
#	client CHAR(8)
#);

