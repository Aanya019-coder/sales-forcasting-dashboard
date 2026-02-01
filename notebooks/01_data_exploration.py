import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv('data/sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("="*60)
print("DATASET INFORMATION")
print("="*60)
print(df.info())

print("\n" + "="*60)
print("FIRST FEW ROWS")
print("="*60)
print(df.head(10))

print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("CATEGORY-WISE SUMMARY")
print("="*60)
category_summary = df.groupby('Category').agg({
    'Revenue': ['sum', 'mean', 'std'],
    'Units_Sold': 'sum',
    'Profit': 'sum'
}).round(2)
print(category_summary)

# Visualize trends
plt.figure(figsize=(14, 7))
for category in df['Category'].unique():
    cat_data = df[df['Category'] == category]
    plt.plot(cat_data['Date'], cat_data['Revenue'], label=category, marker='o', linewidth=2)

plt.title('Revenue Trends by Category Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Revenue ($)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Create assets directory if it doesn't exist
os.makedirs('assets', exist_ok=True)

plt.savefig('assets/revenue_trends.png', dpi=300)
print("\n✅ Chart saved to assets/revenue_trends.png")
plt.show()

print("\n✅ Exploration complete!")
