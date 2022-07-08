import sys
import warnings

import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

# Postgresql
import psycopg2
from config import config
from psycopg2.extensions import register_adapter, AsIs




# Connect to Postgresql database
params = config()
conn = psycopg2.connect(**params)
conn.autocommit = True
cur = conn.cursor()

# Create table
cur.execute("DROP TABLE IF EXISTS emnist;"
            "CREATE TABLE IF NOT EXISTS emnist ("
                "id INT PRIMARY KEY NOT NULL,"
                "client CHAR(8)"
                ")")
cur.execute("CREATE INDEX idx_client ON emnist(client)")
cur.execute("SET search_path = schema1, public;")

for i in range(5):
    client = "asd" + str(i)
    cur.execute("INSERT INTO emnist("
                                "id," 
                                "client)"
                            "VALUES (%s, %s)",
                            (i, 
                            client)
                )


# Close connection
cur.close()
conn.close()