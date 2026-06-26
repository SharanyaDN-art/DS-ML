import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Define directory paths
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data")
graphs_dir = os.path.join(base_dir, "graphs")
output_dir = os.path.join(base_dir, "output")

# Ensure directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Set up logging to both console and file
log_file_path = os.path.join(output_dir, "analysis_output.txt")
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(log_file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

print("Understanding dataset")

file_name = os.path.join(data_dir, "sales_data.csv")
if not os.path.exists(file_name):
    # Fallback to current directory check
    fallback_path = os.path.join(base_dir, "sales_data.csv")
    if os.path.exists(fallback_path):
        file_name = fallback_path
    else:
        print(f"File {file_name} not found")
        exit()

#Reading csv file 
df=pd.read_csv(file_name)
print("Sucessfully loaded")
print(f"shape of the dataset:{df.shape[0]},columns:{df.shape[1]}")

print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Last 5 Rows ---")
print(df.tail())
print("\n--- Summary Statistics ---")
print(df.describe())

#Handling Missing Values
print("\n--- Handling Missing Values ---")
print("Missing values count per column:")
print(df.isnull().sum())
print("\nRows with missing Spending values:")
print(df[df['Spending'].isnull()])

#with using median 
median_age=df['Age'].median()
df['Age'].fillna(median_age,inplace=True)
print(f"\nFilled missing Age with median_age: {median_age}")

#Spending missing values 
print("\nMissing values in Spending after cleaning Age:")
print(df['Spending'].isnull().sum())

#using mean 
mean_spending=df['Spending'].mean()
df['Spending']=df['Spending'].fillna(mean_spending)
print(f"Filled missing Spending with mean_spending: {mean_spending}")

# plotting histogram for spending amount
plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10, color="white",edgecolor='Black')
plt.title("Distribution of Spending")
plt.xlabel("Spending Amount")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "spending_distribution.png"), dpi=150)
print("\nSaved Spending Distribution plot to graphs/spending_distribution.png")
plt.show()

# Calculate correlation only on numeric columns
numeric_df = df.select_dtypes(include=['number'])
correlation = numeric_df.corr()
print("\n--- Correlation Matrix ---")
print(correlation)
print("Plotting Correlation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap="coolwarm",linewidths=1)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "correlation_heatmap.png"), dpi=150)
print("Saved Correlation Heatmap to graphs/correlation_heatmap.png")
plt.show()

#Outlier Detection
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Age'], color='lightgreen')
plt.title("BoxPlot of Customer Age")
plt.xlabel("Age")
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, "age_boxplot.png"), dpi=150)
print("\nSaved Customer Age Boxplot to graphs/age_boxplot.png")
plt.show()

print("\n--- Find the Outliers in Age ---")
Outliers=df[df['Age']>100]
print("Found Outliers(s):")
print(Outliers)

# Save cleaned data to output folder
cleaned_data_path = os.path.join(output_dir, "cleaned_sales_data.csv")
df.to_csv(cleaned_data_path, index=False)
print(f"\nSaved cleaned dataset to {cleaned_data_path}")
