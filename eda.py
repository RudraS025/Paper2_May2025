import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the imputed data
df = pd.read_excel('Paper_May25_imputed.xlsx')

print("Starting EDA plots...")

# 1. See basic info and statistics
print(df.info())
print(df.describe())

# 2. Correlation heatmap (shows how variables are related)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
plt.savefig('correlation_heatmap.png')
plt.close()
print("Correlation heatmap saved as correlation_heatmap.png")

# 3. Distribution plots for all variables (except date)
for col in df.columns:
    if col != 'Months':
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        filename = f"distribution_{col.replace('/', '_or_')}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Distribution plot for {col} saved as {filename}")