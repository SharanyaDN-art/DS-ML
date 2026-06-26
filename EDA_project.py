import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

print("understanding dataset")

file_name="sales_data.csv"
if not os.path.exists(file_name):
    print(f"Error:{file_name}is not found")

df=pd.read_csv(file_name)
print("succesfully loaded")
print(f"Shape of the dataset: Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print(df.head())
print(df.tail())
print(df.describe())

print("handling missing values")
print(df.isnull().sum())

median_age=df['Age'].median()
df['Age']=df['Age'].fillna(median_age)
print(median_age)