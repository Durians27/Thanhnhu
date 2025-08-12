import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Data Collection/Generation
# Generate synthetic monthly sales data over 2 years (24 months) with an upward trend and random noise.
# Introduce null values and duplicates for preprocessing demonstration.
np.random.seed(42)
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=30*i) for i in range(24)]  # 2 years monthly
sales = np.random.normal(10000, 2000, 24) + np.arange(24)*200  # Trend + noise

# Introduce null values (simulate missing data)
sales[3] = np.nan  # Missing value in April 2023
sales[10] = np.nan  # Missing value in November 2023

# Create DataFrame
df = pd.DataFrame({'Date': dates, 'Sales': sales})
df.set_index('Date', inplace=True)

# Add duplicate rows (simulate data entry error)
duplicate_rows = df.iloc[[0, 5]]  # Duplicate January and June 2023
df = pd.concat([df, duplicate_rows]).sort_index()

# Synthetic category and region data for visualizations
categories = ['Electronics', 'Clothing', 'Books', 'Home']
regions = ['North', 'South', 'East', 'West']
category_sales = np.random.normal(5000, 1000, (24, 4))  # Monthly sales per category
region_sales = np.random.normal(3000, 500, (24, 4))  # Monthly sales per region
category_df = pd.DataFrame(category_sales, index=dates, columns=categories)
region_df = pd.DataFrame(region_sales, index=dates, columns=regions)

# Step 2: Data Preprocessing
# 2.1: Handling Null (Missing) Values
print("Checking for null values:")
missing = df.isnull().sum()
print(missing)

# Reason for handling nulls: Missing values can distort statistical analysis and reduce model accuracy.
# For time-series data, forward-fill is appropriate to maintain continuity.
df['Sales'] = df['Sales'].fillna(method='ffill')
print("\nAfter handling nulls:")
print(df.isnull().sum())

# 2.2: Removing Duplicate Records
# Reason for removal: Duplicates bias results by over-representing certain observations, leading to overfitting or skewed forecasts.
print("\nChecking for duplicates:")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
df = df.drop_duplicates()
print(f"Number of rows after removing duplicates: {len(df)}")

# 2.3: Data Normalization (Standardization)
# Reason for standardization: Machine learning models (e.g., ARIMA, neural networks) perform better when features are on a similar scale, preventing large values from dominating.
scaler = StandardScaler()
df['Sales_Standardized'] = scaler.fit_transform(df[['Sales']])
print("\nData after standardization (first 5 rows):")
print(df.head())

# 2.4: Split into Train and Test
train = df.iloc[:-6][['Sales', 'Sales_Standardized']]
test = df.iloc[-6:][['Sales', 'Sales_Standardized']]
print("\nTrain set size:", len(train))
print("Test set size:", len(test))

# Save preprocessed data for further steps
df.to_csv('preprocessed_sales_data.csv')

# Step 3: Data Analysis and Visualization (5 Visualizations)
# Statistical summary
print("\nSales Summary:")
print(df['Sales'].describe())
print("\nLast Month Category Sales:")
print(category_df.iloc[-1])
print("\nLast Month Regional Sales:")
print(region_df.iloc[-1])

# Visualization 1: Monthly Sales Trend
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Sales'], marker='o', label='Actual Sales')
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.grid(True)
plt.show()  # In Colab/Jupyter, displays the plot

# Visualization 2: Sales by Product Category (Stacked Area)
plt.figure(figsize=(10, 5))
plt.stackplot(category_df.index, category_df.T, labels=categories, alpha=0.6)
plt.title('Sales by Product Category')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Visualization 3: Regional Sales Share (Pie Chart for Last Month)
last_month = region_df.iloc[-1]
plt.figure(figsize=(8, 8))
plt.pie(last_month, labels=regions, autopct='%1.1f%%', startangle=90)
plt.title('Regional Sales Share (Last Month)')
plt.show()

# Step 4: Model Training
model = ARIMA(train['Sales'], order=(1,1,1))
model_fit = model.fit()
print("\nARIMA Model Summary:")
print(model_fit.summary())

# Step 5: Model Evaluation and Forecasting
pred = model_fit.forecast(steps=6)
mae = mean_absolute_error(test['Sales'], pred)
rmse = np.sqrt(mean_squared_error(test['Sales'], pred))
print(f"\nMAE: {mae}, RMSE: {rmse}")

# Forecast next 6 months
forecast_steps = 6
forecast = model_fit.forecast(steps=forecast_steps)
forecast_dates = [df.index[-1] + timedelta(days=30*(i+1)) for i in range(forecast_steps)]
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Sales': forecast})
print("\nForecasted Sales:")
print(forecast_df)

# Visualization 4: Actual vs. Forecasted Sales
plt.figure(figsize=(10, 5))
plt.plot(test.index, test['Sales'], marker='o', label='Actual Sales')
plt.plot(test.index, pred, marker='x', label='Forecasted Sales', linestyle='--')
plt.title('Actual vs. Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.grid(True)
plt.show()

# Visualization 5: Sales Distribution (Histogram with KDE)
plt.figure(figsize=(10, 5))
sns.histplot(df['Sales'], kde=True, bins=10)
plt.title('Sales Distribution')
plt.xlabel('Sales ($)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()