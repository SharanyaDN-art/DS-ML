import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Understanding dataset")

# Try loading data.csv (tab-separated) or sales_data.csv (comma-separated)
file_name = os.path.join(os.path.dirname(__file__), "data.csv")
if os.path.exists(file_name):
    df = pd.read_csv(file_name, sep='\t')
else:
    file_name = os.path.join(os.path.dirname(__file__), "sales_data.csv")
    if not os.path.exists(file_name):
        file_name = os.path.join(os.path.dirname(__file__), "data", "sales_data.csv")
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
    else:
        print("Error: Could not find data.csv or sales_data.csv")
        exit()
print("Sucessfully loaded")
print(f"shape of the dataset:{df.shape[0]},columns:{df.shape[1]}")

print(df.head())
print(df.tail())
print(df.describe())

#Handling Missing Values
print("Handling missing values")
print(df.isnull().sum())
print(df[df['Spending'].isnull()])

#with using median 
median_age=df['Age'].median()
df['Age'].fillna(median_age,inplace=True)
print(f"median_age:{median_age}")

#Spending missing values 
print("Spending missing values")
print(df['Spending'].isnull().sum())
print(df[df['Spending'].isnull()])

#using mean 
mean_spending=df['Spending'].mean()
df['Spending']=df['Spending'].fillna(mean_spending)
print(mean_spending)

# plotting histogram for spending amount
plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10, color="white",edgecolor='Black')
plt.title("Distribution of Spending")
plt.xlabel("Spending Amount")
plt.ylabel("Number of Customers")
plt.show()

# Calculate correlation only on numeric columns
numeric_df = df.select_dtypes(include=['number'])
correlation = numeric_df.corr()
print(correlation)
print("Plotting Correlation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap="coolwarm",linewidths=1)
plt.title("Correlation Heatmap")
plt.show()

#Outlier Detection
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Age'], color='lightgreen')
plt.title("BoxPlot of Customer Age")
plt.xlabel("Age")
plt.show()

print("Find the Outliers in age")
Outliers=df[df['Age']>100]
print("Found Outliers(s):")
print(Outliers)