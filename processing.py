import csv
import sys
import os
import random
import sys
from tqdm.notebook import tqdm
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

graphs_folder = "graphs"
if not os.path.exists(graphs_folder):
  os.makedirs(graphs_folder)

con = sqlite3.connect('./transaction_data.db')
cur = con.cursor()

def describe_table():
    cur.execute("select sql from sqlite_master where type = 'table' and name = 'transaction_data';")
    print(cur.fetchall()[0][0])

def get_first_record():
    cur.execute("select * from transaction_data LIMIT 1;")
    print(cur.fetchall()[0])

# (1, 1, 'PAYMENT', '9839.640', 'C1231006815', '170136.000', '160296.360', 'M1979787155', '0.000', '9839.640')

def export_csv(filename='./transactions.csv'):
    f = open(filename,'w')
    f.write('timestamp,type,amount,nameOrig,oldbalanceOrig,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest\n')
    cur.execute("select * from transaction_data;")
    data = cur.fetchall()
    for line in tqdm(data):
        f.write(str(line[1])+','+line[2]+','+format(float(line[3]),'.2f')+','+line[4]+','+format(float(line[5]),'.2f')+','+format(float(line[6]),'.2f')+','+line[7]+','+format(float(line[8]),'.2f')+','+format(float(line[9]),'.2f')+'\n')

def string_amount_to_int(i):
    if "E" in i:
        return float(i)*1000
    else:
        major, decimals = i.split(".")
#         assert len(decimals) <= 3, "Found a number with more than 3 decimal points:"+str(i)
        decimals = decimals[:3].ljust(3, "0")
        return int(major+decimals)

def read_to_pandas():
    cur.execute("select * from transaction_data;")
#     cur.execute("select * from transaction_data LIMIT 10000;")
    columns = 'timestamp,type,amount,nameOrig,oldbalanceOrig,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest'.split(",")
    columns.insert(0, "tx_index")
    df = pd.DataFrame(data=cur.fetchall(), columns=columns)
    for col in ["tx_index", "timestamp", "amount", "oldbalanceOrig", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        if col in ["amount", "oldbalanceOrig", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        #Converting to an integer by factor of 1000!
            df[col] = df[col].apply(string_amount_to_int)
        else:
            df[col] = pd.to_numeric(df[col])
    return df

if __main__ == "__name__":
    if not os.path.exists("df_transactions.csv"):
        describe_table()
        get_first_record()
        df = read_to_pandas()
        df.set_index("tx_index", inplace=True, drop=False)
        df.to_csv("df_transactions.csv")
        print(df.dtypes)
        print(df)
    else:
        df = pandas.read_csv("df_transactions.csv")
    
    # Doing the actual processing.
    # Describing the general dataset:
    description = df.describe()
    print(f"There are {df.shape[0]} transactions in the dataset.")
    mean_transaction = description.loc["mean", "amount"]
    median_transaction =  df["amount"].median()
    total_transaction =  df["amount"].sum()
    print(f"The mean transaction has a value of {mean_transaction}, median transaction is {median_transaction} and a total of {total_transaction} was transacted.")
    print(description)

    # Graphing histogram account balances.
    fig, axs = plt.subplots(2, sharex=True)
    df["oldbalanceOrig"].divide(1000).hist(ax=axs[0], bins=50)
    axs[0].set_yscale('log')
    axs[0].set_ylabel("Number of accounts")
    axs[0].set_xlabel("Account balance old")
    df["newbalanceOrig"].divide(1000).hist(ax=axs[1], bins=50)
    axs[1].set_yscale('log')
    axs[1].set_ylabel("Number of accounts")
    axs[1].set_xlabel("Account balance new")
    fig = plt.gcf()
    fig.set_size_inches(11, 5)
    plt.savefig(os.path.join(graphs_folder, "Histogram_account_balances_stacked.pdf"), bbox_inches='tight')

    # Graphing transaction amounts:
    ax = df["amount"].hist(bins=50)
    ax.set_yscale('log')
    ax.set_ylabel("Number of transactions")
    ax.set_xlabel("Transaction amounts")
    fig = plt.gcf()
    fig.set_size_inches(11, 3)
    plt.savefig(os.path.join(graphs_folder, "Histogram_Transaction_amounts.pdf"), bbox_inches='tight')

    # Graphing amount distribution:
    account_balances = df["newbalanceOrig"].copy()
    account_balances.sort_values(inplace=True)
    account_balances.index = list(range(account_balances.shape[0]))
    ax = account_balances.plot()
    ax.grid()
    ax.set_ylabel("Account balance")
    ax.set_xlabel("Index")
    # ax.set_yscale('log')
    fig = plt.gcf()
    fig.set_size_inches(11, 4)
    plt.savefig(os.path.join(graphs_folder, "Account_balance_distribution.pdf"), bbox_inches='tight')
    richest_balance = account_balances.iloc[-1]
    print(f"Richest account has {richest_balance}")
    top_to_show = 0.01
    top_index = int(account_balances.shape[0]*(1-top_to_show))
    portion_of_wealth = account_balances.iloc[top_index:].sum()/account_balances.sum()
    print(f"The top {top_to_show*100}% of addresses own {portion_of_wealth*100}% of the wealth")

    # Checking for large transactions:
    large_amount = 1000000
    large_transactions = df[df["amount"] > large_amount]
    print(f"There are {large_transactions.shape[0]} transactions with an amount larger than {large_amount}.")
    print(large_transactions)

    # Checking for small transactions:
    small_amount = 10
    small_transactions = df[df["amount"] <= small_amount]
    print(f"There are {small_transactions.shape[0]} transactions with an amount smaller than {small_amount}.")
    print(small_transactions)

    # Checking for unusually many transactions:
    amount_received_per_nameDest = df.groupby("nameDest")["amount"].agg(['mean', 'count', 'sum'])
    amount_received_per_nameDest.sort_values("count", inplace=True, ascending=False)
    top_address_transaction_count = amount_received_per_nameDest.iloc[0]["count"]
    top_address_transaction_sum = amount_received_per_nameDest.iloc[0]["sum"]
    print(f"The top account receives {int(top_address_transaction_count)} which corresponds to a total transaction amount received of {top_address_transaction_sum}.")
    print(amount_received_per_nameDest)

    # 
    print("Middle man addresses might be malicious if they often receive amounts and then immediately send it onwards.")
    print("This could beamount_sent_per_nameOrig identified by comparing the total amount received by an address to the total amount being sent by an address.")
    amount_sent_per_nameOrig = df.groupby("nameOrig")["amount"].agg(['mean', 'count', 'sum'])
    amount_sent_per_nameOrig.sort_values("count", inplace=True, ascending=False)
    print(amount_sent_per_nameOrig)