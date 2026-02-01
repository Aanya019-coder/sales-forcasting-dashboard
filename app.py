import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import pickle
from datetime import datetime
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data
def load_sales_data():
    """Load historical sales data"""
    df = pd.read_csv('data/sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_inventory_recommendations():
    """Load inventory optimization recommendations"""
    return pd.read_csv('data/inventory_recommendations.csv')

@st.cache_resource
def load_forecast_model(category):
    """Load trained Prophet model for a category"""
    model_path = f'models/forecast_{category}.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

# ============================================================
# LOAD DATA
# ============================================================

try:
    df = load_sales_data()
    recommendations = load_inventory_recommendations()
except FileNotFoundError:
    st.error("⚠️ Data files not found. Please run the setup scripts first:")
    st.code("""
    python src/generate_data.py
    python src/forecast_model.py
    python src/inventory_optimizer.py
    """)
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.markdown('<h1 class="main-header">🚀 Sales Forecasting & Inventory Optimization Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Powered by Machine Learning | Built for Business Decision-Making**")
st.markdown("---")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("⚙️ Dashboard Controls")
st.sidebar.markdown("---")

# Category selection
selected_category = st.sidebar.selectbox(
    "📦 Select Product Category",
    df['Category'].unique(),
    help="Choose a product category to view detailed analytics"
)

# Date range filter
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Date Range Filter")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Forecast period
st.sidebar.markdown("---")
forecast_months = st.sidebar.slider(
    "🔮 Forecast Period (Months)",
    min_value=3,
    max_value=12,
    value=6,
    help="Number of months to forecast ahead"
)

# About section
st.sidebar.markdown("---")
st.sidebar.info("""
**About This Dashboard:**

This tool uses Prophet (Facebook's time-series forecasting algorithm) to:
- Predict future sales trends
- Optimize inventory levels
- Minimize holding costs
- Prevent stockouts

**Tech Stack:**
- Python | Prophet | Streamlit
- Plotly | Pandas | Scikit-learn
""")

# ============================================================
# FILTER DATA
# ============================================================

# Filter by category
cat_data = df[df['Category'] == selected_category].copy()

# Filter by date range
if len(date_range) == 2:
    mask = (cat_data['Date'] >= pd.to_datetime(date_range[0])) & \
           (cat_data['Date'] <= pd.to_datetime(date_range[1]))
    cat_data = cat_data[mask]

# ============================================================
# SECTION 1: KEY PERFORMANCE INDICATORS
# ============================================================

st.header("📈 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = cat_data['Revenue'].sum()
    st.metric(
        label="💰 Total Revenue",
        value=f"${total_revenue:,.0f}",
        delta="Historical"
    )

with col2:
    total_units = cat_data['Units_Sold'].sum()
    st.metric(
        label="📦 Units Sold",
        value=f"{total_units:,.0f}",
        delta="All Time"
    )

with col3:
    total_profit = cat_data['Profit'].sum()
    profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0
    st.metric(
        label="💵 Total Profit",
        value=f"${total_profit:,.0f}",
        delta=f"{profit_margin:.1f}% margin"
    )

with col4:
    avg_revenue = cat_data['Revenue'].mean()
    st.metric(
        label="📊 Avg Monthly Revenue",
        value=f"${avg_revenue:,.0f}",
        delta="Per Month"
    )

st.markdown("---")

# ============================================================
# SECTION 2: HISTORICAL TRENDS
# ============================================================

st.header("📊 Historical Sales Trends")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Revenue Trend", "Units Sold", "Profit Analysis"])

with tab1:
    fig_revenue = px.line(
        cat_data,
        x='Date',
        y='Revenue',
        title=f'{selected_category} - Revenue Over Time',
        markers=True,
        template='plotly_white'
    )
    fig_revenue.update_traces(line_color='#1f77b4', line_width=3)
    fig_revenue.update_layout(
        height=450,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Revenue ($)"
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

with tab2:
    fig_units = px.area(
        cat_data,
        x='Date',
        y='Units_Sold',
        title=f'{selected_category} - Units Sold Over Time',
        template='plotly_white'
    )
    fig_units.update_traces(fillcolor='rgba(31, 119, 180, 0.3)', line_color='#1f77b4')
    fig_units.update_layout(
        height=450,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Units Sold"
    )
    st.plotly_chart(fig_units, use_container_width=True)

with tab3:
    fig_profit = px.bar(
        cat_data,
        x='Date',
        y='Profit',
        title=f'{selected_category} - Monthly Profit',
        template='plotly_white',
        color='Profit',
        color_continuous_scale='RdYlGn'
    )
    fig_profit.update_layout(
        height=450,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Profit ($)"
    )
    st.plotly_chart(fig_profit, use_container_width=True)

st.markdown("---")

# ============================================================
# SECTION 3: SALES FORECAST
# ============================================================

st.header(f"🔮 Sales Forecast (Next {forecast_months} Months)")

# Load model and generate forecast
model = load_forecast_model(selected_category)

if model is not None:
    # Generate forecast
    future = model.make_future_dataframe(periods=forecast_months, freq='M')
    forecast = model.predict(future)
    
    # Create forecast visualization
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=cat_data['Date'],
        y=cat_data['Revenue'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Forecast
    future_forecast = forecast.tail(forecast_months)
    fig_forecast.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat'],
        mode='lines+markers',
        name='Forecasted Sales',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Confidence interval (upper bound)
    fig_forecast.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(255,127,14,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Confidence interval (lower bound)
    fig_forecast.add_trace(go.Scatter(
        x=future_forecast['ds'],
        y=future_forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255,127,14,0)'),
        name='95% Confidence Interval',
        fillcolor='rgba(255,127,14,0.2)'
    ))
    
    fig_forecast.update_layout(
        title=f'{selected_category} - Sales Forecast with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        height=500,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Forecast table
    with st.expander("📋 View Detailed Forecast Data"):
        forecast_table = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_table.columns = ['Date', 'Predicted Revenue', 'Lower Bound', 'Upper Bound']
        forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
        forecast_table['Predicted Revenue'] = forecast_table['Predicted Revenue'].apply(lambda x: f"${x:,.0f}")
        forecast_table['Lower Bound'] = forecast_table['Lower Bound'].apply(lambda x: f"${x:,.0f}")
        forecast_table['Upper Bound'] = forecast_table['Upper Bound'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(forecast_table, use_container_width=True)
    
else:
    st.error(f"⚠️ Forecast model not found for {selected_category}. Please run `python src/forecast_model.py` first.")

st.markdown("---")

# ============================================================
# SECTION 4: INVENTORY OPTIMIZATION
# ============================================================

st.header("📦 Inventory Optimization Recommendations")

# Get recommendations for selected category
cat_rec = recommendations[recommendations['Category'] == selected_category]

if not cat_rec.empty:
    cat_rec = cat_rec.iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Recommended Stock Levels")
        
        st.metric(
            label="🛡️ Safety Stock",
            value=f"{cat_rec['Safety_Stock']:,.0f} units",
            help="Buffer stock to prevent stockouts"
        )
        
        st.metric(
            label="🔔 Reorder Point",
            value=f"{cat_rec['Reorder_Point']:,.0f} units",
            help="Stock level that triggers new order"
        )
        
        st.metric(
            label="📦 Optimal Order Quantity",
            value=f"{cat_rec['Optimal_Order_Quantity']:,.0f} units",
            help="Recommended quantity per order"
        )
    
    with col2:
        st.subheader("💰 Expected Financial Impact")
        
        st.metric(
            label="📊 Forecasted Monthly Revenue",
            value=f"${cat_rec['Estimated_Monthly_Revenue']:,.0f}",
            help="Expected revenue for next month"
        )
        
        st.metric(
            label="📈 Avg Monthly Sales",
            value=f"{cat_rec['Avg_Monthly_Sales_Units']:,.0f} units",
            help="Average units sold per month"
        )
        
        monthly_savings = cat_rec.get('Monthly_Cost_Savings', 0)
        annual_savings = cat_rec.get('Annual_Cost_Savings', 0)
        
        st.metric(
            label="💵 Potential Monthly Savings",
            value=f"${monthly_savings:,.0f}",
            delta=f"${annual_savings:,.0f}/year",
            help="Cost savings from optimized inventory"
        )
    
    # Inventory visualization
    st.subheader("📊 Inventory Level Comparison")
    
    current_stock = cat_data['Stock_Level'].mean()
    optimal_stock = cat_rec['Safety_Stock']
    
    comparison_df = pd.DataFrame({
        'Stock Type': ['Current Practice', 'Recommended Optimal'],
        'Units': [current_stock, optimal_stock]
    })
    
    fig_inventory = px.bar(
        comparison_df,
        x='Stock Type',
        y='Units',
        title='Current vs Optimal Stock Levels',
        color='Stock Type',
        color_discrete_map={
            'Current Practice': '#ff7f0e',
            'Recommended Optimal': '#2ca02c'
        },
        text='Units',
        template='plotly_white'
    )
    fig_inventory.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig_inventory.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_inventory, use_container_width=True)
    
else:
    st.error(f"⚠️ No inventory recommendations found for {selected_category}")

st.markdown("---")

# ============================================================
# SECTION 5: ALL CATEGORIES COMPARISON
# ============================================================

st.header("🏆 Category Performance Comparison")

# Aggregate data by category
category_summary = df.groupby('Category').agg({
    'Revenue': 'sum',
    'Units_Sold': 'sum',
    'Profit': 'sum'
}).reset_index()

# Add profit margin
category_summary['Profit_Margin'] = (category_summary['Profit'] / category_summary['Revenue'] * 100).round(2)

# Create comparison tabs
tab1, tab2, tab3 = st.tabs(["Revenue Comparison", "Units Sold", "Profitability"])

with tab1:
    fig_cat_revenue = px.bar(
        category_summary,
        x='Category',
        y='Revenue',
        title='Total Revenue by Category',
        color='Revenue',
        color_continuous_scale='Blues',
        text='Revenue',
        template='plotly_white'
    )
    fig_cat_revenue.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_cat_revenue.update_layout(height=450)
    st.plotly_chart(fig_cat_revenue, use_container_width=True)

with tab2:
    fig_cat_units = px.pie(
        category_summary,
        values='Units_Sold',
        names='Category',
        title='Units Sold Distribution',
        hole=0.4,
        template='plotly_white'
    )
    fig_cat_units.update_traces(textposition='inside', textinfo='percent+label')
    fig_cat_units.update_layout(height=450)
    st.plotly_chart(fig_cat_units, use_container_width=True)

with tab3:
    fig_cat_margin = px.bar(
        category_summary.sort_values('Profit_Margin', ascending=False),
        x='Category',
        y='Profit_Margin',
        title='Profit Margin by Category (%)',
        color='Profit_Margin',
        color_continuous_scale='RdYlGn',
        text='Profit_Margin',
        template='plotly_white'
    )
    fig_cat_margin.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_cat_margin.update_layout(height=450, yaxis_title="Profit Margin (%)")
    st.plotly_chart(fig_cat_margin, use_container_width=True)

# Summary table
st.subheader("📊 Summary Table")
summary_display = category_summary.copy()
summary_display['Revenue'] = summary_display['Revenue'].apply(lambda x: f"${x:,.0f}")
summary_display['Units_Sold'] = summary_display['Units_Sold'].apply(lambda x: f"{x:,.0f}")
summary_display['Profit'] = summary_display['Profit'].apply(lambda x: f"${x:,.0f}")
summary_display['Profit_Margin'] = summary_display['Profit_Margin'].apply(lambda x: f"{x:.2f}%")
st.dataframe(summary_display, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================
# FOOTER - BUSINESS INSIGHTS
# ============================================================

st.header("💡 Key Business Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🎯 What This Dashboard Enables:**
    - **Accurate Forecasting**: Predict sales 3-12 months ahead with 95% confidence intervals
    - **Cost Optimization**: Reduce inventory holding costs by 15-25%
    - **Risk Mitigation**: Prevent stockouts and overstock situations
    - **Data-Driven Decisions**: Replace gut feeling with statistical models
    
    **📈 How to Use:**
    1. Select product category from sidebar
    2. Review historical trends and KPIs
    3. Analyze forecast predictions
    4. Implement inventory recommendations
    5. Monitor actual vs predicted performance
    """)

with col2:
    total_potential_savings = recommendations['Annual_Cost_Savings'].sum() if 'Annual_Cost_Savings' in recommendations.columns else 0
    
    st.success(f"""
    **💰 Potential Annual Savings**
    
    By implementing these inventory optimization recommendations across all categories:
    
    **${total_potential_savings:,.0f} per year**
    
    This represents a significant improvement in working capital efficiency and cash flow management.
    """)
    
    st.info("""
    **🔄 Next Steps:**
    1. Validate forecasts with actual sales data monthly
    2. Adjust safety stock based on service level requirements
    3. Integrate with ERP/procurement systems
    4. Set up automated alerts for reorder points
    5. Conduct quarterly model retraining
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Sales Forecasting & Inventory Optimization Dashboard</strong></p>
    <p>Built with ❤️ using Python | Streamlit | Prophet | Plotly</p>
    <p>📧 Contact | 💼 LinkedIn | 🐙 GitHub</p>
</div>
""", unsafe_allow_html=True)
