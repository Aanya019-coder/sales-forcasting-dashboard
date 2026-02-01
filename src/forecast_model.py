import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os

def train_forecast_model(category):
    """
    Train Prophet model for a specific product category
    
    Args:
        category (str): Product category name
    
    Returns:
        Prophet: Trained model
    """
    
    print(f"\n{'='*60}")
    print(f"Training model for: {category}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv('data/sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for specific category
    cat_data = df[df['Category'] == category][['Date', 'Revenue']].copy()
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    cat_data.columns = ['ds', 'y']
    
    print(f"Training data points: {len(cat_data)}")
    print(f"Date range: {cat_data['ds'].min()} to {cat_data['ds'].max()}")
    print(f"Revenue range: ${cat_data['y'].min():,.0f} to ${cat_data['y'].max():,.0f}")
    
    # Initialize and train model
    model = Prophet(
        yearly_seasonality=True,      # Capture yearly patterns
        weekly_seasonality=False,      # Not needed for monthly data
        daily_seasonality=False,       # Not needed for monthly data
        changepoint_prior_scale=0.05,  # Flexibility of trend changes
        seasonality_mode='multiplicative'  # Seasonality scales with trend
    )
    
    model.fit(cat_data)
    
    # Evaluate on historical data
    forecast = model.predict(cat_data)
    mae = mean_absolute_error(cat_data['y'], forecast['yhat'])
    rmse = np.sqrt(mean_squared_error(cat_data['y'], forecast['yhat']))
    mape = np.mean(np.abs((cat_data['y'] - forecast['yhat']) / cat_data['y'])) * 100
    
    print(f"\nModel Performance Metrics:")
    print(f"  MAE (Mean Absolute Error): ${mae:,.0f}")
    print(f"  RMSE (Root Mean Squared Error): ${rmse:,.0f}")
    print(f"  MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/forecast_{category}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to: {model_path}")
    
    return model

def predict_future_sales(category, periods=6):
    """
    Predict future sales for next N months
    
    Args:
        category (str): Product category name
        periods (int): Number of months to forecast ahead
    
    Returns:
        DataFrame: Forecast with predictions and confidence intervals
    """
    
    # Load trained model
    model_path = f'models/forecast_{category}.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Create future dates dataframe
    future = model.make_future_dataframe(periods=periods, freq='M')
    
    # Generate predictions
    forecast = model.predict(future)
    
    # Return relevant columns
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SALES FORECASTING MODEL TRAINING")
    print("="*60)
    
    # Load categories
    df = pd.read_csv('data/sales_data.csv')
    categories = df['Category'].unique()
    
    print(f"\nFound {len(categories)} categories: {', '.join(categories)}")
    
    # Train model for each category
    for category in categories:
        train_forecast_model(category)
    
    print("\n" + "="*60)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    
    # Test prediction
    print("\nTesting prediction for Electronics...")
    test_forecast = predict_future_sales('Electronics', periods=6)
    print("\nNext 6 months forecast:")
    print(test_forecast)
