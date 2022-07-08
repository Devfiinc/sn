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




def main(argv):
    # Connect to Postgresql database
    params = config()
    conn = psycopg2.connect(**params)
    conn.autocommit = True
    cur = conn.cursor()

    query = "SELECT * FROM sn"
    df = pd.read_sql_query(query, conn)

    print(df.head(5))

    # Close connection
    cur.close()
    conn.close()









    # Non-private version of Logistic Regression
    print()
    print()
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("                            Logistics Regression (non-private)                             ")
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print()
    
    _ = df['class'].value_counts().sort_index().plot(kind='bar', title = 'Class distribution')

    seed = 1

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['class'],axis=1), df['class'], test_size=0.2, stratify = df['class'], random_state = seed)

    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)
    X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)




    np.random.seed(seed)

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter = 500)
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    print(classification_report(y_test, y_pred))

    y_probs = clf.predict_proba(X_test_std)
    roc_score = roc_auc_score(y_test, y_probs, multi_class='ovo')
    print('ROC AUC Score: %f' %roc_score)
    






    # Differentially private version of Logistic Regression
    print()
    print()
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("                              Logistics Regression (private)                               ")
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print()

    np.random.seed(seed)

    from diffprivlib.models import GaussianNB, LogisticRegression # DP version of classifier

    clf = LogisticRegression(epsilon = 3.0, data_norm=7.89)

    clf.fit(X_train_std, y_train)

    y_pred = clf.predict(X_test_std)
    print(classification_report(y_test, y_pred))

    y_probs = clf.predict_proba(X_test_std)
    roc_score = roc_auc_score(y_test, y_probs, multi_class='ovo')
    print('ROC AUC Score: %f' %roc_score)




















    

if __name__ == '__main__':
    main(sys.argv)
