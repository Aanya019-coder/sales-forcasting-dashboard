import pandas as pd
import pickle
import os

def calculate_optimal_inventory(category, forecast_df, safety_stock_days=30, lead_time_days=14):
    """
    Calculate optimal inventory levels based on sales forecast
    
    Args:
        category (str): Product category
        forecast_df (DataFrame): Sales forecast dataframe
        safety_stock_days (int): Days of safety buffer stock
        lead_time_days (int): Days to receive new stock
    
    Returns:
        dict: Inventory recommendations
    """
    
    print(f"\n{'='*60}")
    print(f"Calculating inventory for: {category}")
    print(f"{'='*60}")
    
    # Get average monthly sales from forecast (next 6 months)
    future_forecast = forecast_df.tail(6)
    avg_monthly_sales = future_forecast['yhat'].mean()
    avg_daily_sales = avg_monthly_sales / 30
    
    print(f"Forecasted avg monthly sales: ${avg_monthly_sales:,.0f}")
    print(f"Forecasted avg daily sales: ${avg_daily_sales:,.0f}")
    
    # Calculate safety stock (buffer to prevent stockouts)
    safety_stock = int(avg_daily_sales * safety_stock_days)
    print(f"Safety stock ({safety_stock_days} days): {safety_stock:,} units")
    
    # Calculate reorder point (when to order new stock)
    # ROP = (Average daily sales × Lead time) + Safety stock
    reorder_point = int((avg_daily_sales * lead_time_days) + safety_stock)
    print(f"Reorder point (trigger at): {reorder_point:,} units")
    
    # Calculate optimal order quantity
    # Simplified EOQ: Order enough for 2 months demand
    monthly_demand = int(avg_monthly_sales)
    order_quantity = monthly_demand * 2
    print(f"Optimal order quantity: {order_quantity:,} units")
    
    # Calculate potential cost savings
    # Assume current practice is holding 3 months of stock
    current_stock_level = monthly_demand * 3
    optimal_stock_level = safety_stock + (monthly_demand / 2)  # Average inventory
    excess_stock = current_stock_level - optimal_stock_level
    
    # Assuming $5 holding cost per unit per month
    holding_cost_per_unit = 5
    monthly_savings = excess_stock * holding_cost_per_unit
    annual_savings = monthly_savings * 12
    
    print(f"\nCost Analysis:")
    print(f"  Current avg stock: {current_stock_level:,.0f} units")
    print(f"  Optimal avg stock: {optimal_stock_level:,.0f} units")
    print(f"  Excess stock: {excess_stock:,.0f} units")
    print(f"  Monthly savings: ${monthly_savings:,.0f}")
    print(f"  Annual savings: ${annual_savings:,.0f}")
    
    return {
        'Category': category,
        'Avg_Monthly_Sales_Units': monthly_demand,
        'Avg_Monthly_Revenue': int(avg_monthly_sales),
        'Safety_Stock': safety_stock,
        'Reorder_Point': reorder_point,
        'Optimal_Order_Quantity': order_quantity,
        'Estimated_Monthly_Revenue': int(avg_monthly_sales),
        'Monthly_Cost_Savings': int(monthly_savings),
        'Annual_Cost_Savings': int(annual_savings)
    }

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("INVENTORY OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/sales_data.csv')
    categories = df['Category'].unique()
    
    recommendations = []
    
    # Process each category
    for category in categories:
        # Load trained model
        model_path = f'models/forecast_{category}.pkl'
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found for {category}. Run forecast_model.py first.")
            continue
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Generate 6-month forecast
        future = model.make_future_dataframe(periods=6, freq='M')
        forecast = model.predict(future)
        
        # Calculate inventory recommendations
        rec = calculate_optimal_inventory(category, forecast)
        recommendations.append(rec)
    
    # Save recommendations
    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv('data/inventory_recommendations.csv', index=False)
    
    print("\n" + "="*60)
    print("INVENTORY RECOMMENDATIONS SUMMARY")
    print("="*60)
    print(rec_df.to_string(index=False))
    
    print("\n" + "="*60)
    total_annual_savings = rec_df['Annual_Cost_Savings'].sum()
    print(f"💰 TOTAL POTENTIAL ANNUAL SAVINGS: ${total_annual_savings:,.0f}")
    print("="*60)
    
    print("\n✅ Recommendations saved to: data/inventory_recommendations.csv")
