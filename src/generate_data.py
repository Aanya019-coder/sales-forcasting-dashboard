import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Generate synthetic retail data
np.random.seed(42)

# Date range: 3 years of monthly data
start_date = datetime(2021, 1, 1)
dates = [start_date + timedelta(days=30*i) for i in range(36)]

# Product categories
categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']

data = []
for date in dates:
    for category in categories:
        # Create realistic sales patterns with seasonality
        base_sales = np.random.randint(5000, 15000)
        
        # Add seasonality (higher in Nov-Dec)
        if date.month in [11, 12]:
            seasonal_boost = 1.5
        elif date.month in [6, 7]:
            seasonal_boost = 1.2
        else:
            seasonal_boost = 1.0
        
        # Add trend (growing over time)
        trend = 1 + (dates.index(date) * 0.02)
        
        # Calculate final sales
        sales = int(base_sales * seasonal_boost * trend * np.random.uniform(0.9, 1.1))
        
        # Calculate cost and profit
        cost_per_unit = np.random.randint(20, 100)
        units_sold = sales // cost_per_unit
        revenue = sales
        cost = units_sold * cost_per_unit
        profit = revenue - cost
        
        data.append({
            'Date': date,
            'Category': category,
            'Revenue': revenue,
            'Units_Sold': units_sold,
            'Cost': cost,
            'Profit': profit,
            'Stock_Level': np.random.randint(100, 500)
        })

df = pd.DataFrame(data)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

df.to_csv('data/sales_data.csv', index=False)
print("✅ Data generated successfully!")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
