import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff6b6b;
    }
    .stMetric > div > div > div > div {
        color: #2e86ab;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA GENERATION AND CACHING
# =============================================================================
@st.cache_data
def generate_sample_data():
    """Generate sample sales data"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    
    data = []
    for date in dates:
        for region in ['North', 'South', 'East', 'West']:
            for product in ['Product A', 'Product B', 'Product C']:
                sales = np.random.normal(1000, 300)
                if sales < 0:
                    sales = abs(sales)
                
                data.append({
                    'date': date,
                    'region': region,
                    'product': product,
                    'sales': sales,
                    'units_sold': int(sales / np.random.uniform(10, 50)),
                    'customer_count': np.random.randint(10, 100)
                })
    
    return pd.DataFrame(data)

@st.cache_data
def load_kpi_data(df, start_date, end_date, selected_regions, selected_products):
    """Calculate KPIs based on filters"""
    filtered_df = df[
        (df['date'] >= start_date) & 
        (df['date'] <= end_date) &
        (df['region'].isin(selected_regions)) &
        (df['product'].isin(selected_products))
    ]
    
    total_sales = filtered_df['sales'].sum()
    total_units = filtered_df['units_sold'].sum()
    avg_sales_per_day = filtered_df.groupby('date')['sales'].sum().mean()
    total_customers = filtered_df['customer_count'].sum()
    
    return {
        'total_sales': total_sales,
        'total_units': total_units,
        'avg_daily_sales': avg_sales_per_day,
        'total_customers': total_customers,
        'filtered_df': filtered_df
    }

# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main():
    # Title and description
    st.title("📊 Sales Analytics Dashboard")
    st.markdown("Real-time sales performance monitoring and analysis")
    
    # Load data
    with st.spinner("Loading data..."):
        df = generate_sample_data()
    
    # ==========================================================================
    # SIDEBAR CONTROLS
    # ==========================================================================
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Date range selector
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2024, 1, 1),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime(2024, 12, 31),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )
    
    # Region filter
    st.sidebar.subheader("Regions")
    all_regions = df['region'].unique().tolist()
    selected_regions = st.sidebar.multiselect(
        "Select Regions:",
        all_regions,
        default=all_regions
    )
    
    # Product filter
    st.sidebar.subheader("Products")
    all_products = df['product'].unique().tolist()
    selected_products = st.sidebar.multiselect(
        "Select Products:",
        all_products,
        default=all_products
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # ==========================================================================
    # MAIN CONTENT AREA
    # ==========================================================================
    
    # Validate selections
    if not selected_regions:
        st.error("Please select at least one region")
        return
    if not selected_products:
        st.error("Please select at least one product")
        return
    
    # Load KPI data
    kpi_data = load_kpi_data(df, start_date, end_date, selected_regions, selected_products)
    filtered_df = kpi_data['filtered_df']
    
    # ==========================================================================
    # KPI METRICS ROW
    # ==========================================================================
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Sales",
            value=f"${kpi_data['total_sales']:,.0f}",
            delta=f"+{np.random.uniform(5, 15):.1f}%"
        )
    
    with col2:
        st.metric(
            label="📦 Units Sold",
            value=f"{kpi_data['total_units']:,}",
            delta=f"+{np.random.uniform(-5, 10):.1f}%"
        )
    
    with col3:
        st.metric(
            label="📊 Avg Daily Sales",
            value=f"${kpi_data['avg_daily_sales']:,.0f}",
            delta=f"{np.random.uniform(-10, 5):.1f}%"
        )
    
    with col4:
        st.metric(
            label="👥 Total Customers",
            value=f"{kpi_data['total_customers']:,}",
            delta=f"+{np.random.uniform(0, 8):.1f}%"
        )
    
    st.markdown("---")
    
    # ==========================================================================
    # CHARTS ROW 1
    # ==========================================================================
    st.subheader("📊 Sales Analytics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series chart
        daily_sales = filtered_df.groupby('date')['sales'].sum().reset_index()
        fig_line = px.line(
            daily_sales, 
            x='date', 
            y='sales',
            title="Daily Sales Trend",
            height=400
        )
        fig_line.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col2:
        # Sales by region pie chart
        region_sales = filtered_df.groupby('region')['sales'].sum().reset_index()
        fig_pie = px.pie(
            region_sales,
            values='sales',
            names='region',
            title="Sales by Region",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # ==========================================================================
    # CHARTS ROW 2
    # ==========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        # Product performance bar chart
        product_sales = filtered_df.groupby('product')['sales'].sum().reset_index()
        fig_bar = px.bar(
            product_sales,
            x='product',
            y='sales',
            title="Sales by Product",
            color='sales',
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(
            xaxis_title="Product",
            yaxis_title="Sales ($)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Regional comparison
        region_product = filtered_df.groupby(['region', 'product'])['sales'].sum().reset_index()
        fig_grouped = px.bar(
            region_product,
            x='region',
            y='sales',
            color='product',
            title="Sales by Region and Product",
            barmode='group'
        )
        st.plotly_chart(fig_grouped, use_container_width=True)
    
    # ==========================================================================
    # DETAILED DATA SECTION
    # ==========================================================================
    st.subheader("📋 Detailed Data")
    
    # Tab layout for different views
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Trends", "🔍 Raw Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Performing Regions**")
            top_regions = filtered_df.groupby('region')['sales'].sum().sort_values(ascending=False)
            st.dataframe(top_regions.head().reset_index())
        
        with col2:
            st.write("**Product Performance**")
            product_metrics = filtered_df.groupby('product').agg({
                'sales': 'sum',
                'units_sold': 'sum',
                'customer_count': 'sum'
            }).round(2)
            st.dataframe(product_metrics)
    
    with tab2:
        # Monthly trends
        filtered_df['month'] = filtered_df['date'].dt.to_period('M')
        monthly_trends = filtered_df.groupby(['month', 'region'])['sales'].sum().reset_index()
        monthly_trends['month'] = monthly_trends['month'].astype(str)
        
        fig_trends = px.line(
            monthly_trends,
            x='month',
            y='sales',
            color='region',
            title="Monthly Sales Trends by Region"
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with tab3:
        # Raw data with search and sort
        st.write("**Filter and Search Data**")
        
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("Search in data:")
        with col2:
            sort_column = st.selectbox("Sort by:", filtered_df.columns.tolist())
        
        display_df = filtered_df.copy()
        
        if search_term:
            mask = display_df.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            display_df = display_df[mask]
        
        display_df = display_df.sort_values(sort_column, ascending=False)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # ==========================================================================
    # FOOTER
    # ==========================================================================
    st.markdown("---")
    st.markdown(
        "**Dashboard Info:** "
        f"Showing data from {start_date} to {end_date} | "
        f"Regions: {', '.join(selected_regions)} | "
        f"Products: {', '.join(selected_products)} | "
        f"Total Records: {len(filtered_df):,}"
    )

# =============================================================================
# STREAMLIT APP EXECUTION
# =============================================================================
if __name__ == "__main__":
    main()