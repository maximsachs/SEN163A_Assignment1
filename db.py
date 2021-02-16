#!/usr/bin/python3

import csv
import sys
import os
import random
import sys
from tqdm import tqdm
import sqlite3


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

describe_table()
get_first_record()
export_csv()
