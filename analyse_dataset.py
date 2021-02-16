import pandas 

df = pandas.read_csv("transactions_test.csv")

print(df.groupby("timestamp").count())
