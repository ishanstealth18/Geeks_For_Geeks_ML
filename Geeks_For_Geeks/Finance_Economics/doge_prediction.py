import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

doge_df = pd.read_csv('DOGE-USD.csv')

# convert Date to datetime
doge_df['Date'] = pd.to_datetime(doge_df['Date'])

# separate Year/Month/Day/Hours
doge_df['Year'] = doge_df['Date'].dt.year
doge_df['Month'] = doge_df['Date'].dt.month
doge_df['Day'] = doge_df['Date'].dt.day
doge_df['Hours'] = doge_df['Date'].dt.hour

doge_df.drop('Date',axis=1, inplace=True)

print(doge_df.info())

# check NaN values
print(doge_df.isna().sum())

# check distribution for NaN columns and impute values
nan_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
fig,axes = plt.subplots(2,3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(nan_cols) :
    axes[i].hist(doge_df[col], bins=np.logspace(np.log10(doge_df[col].min()), np.log10(doge_df[col].max()), 20), color='skyblue', edgecolor='black', alpha=0.7)

    # Customize individual subplot details
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, linestyle='--', alpha=0.5)

# Adjust spacing between charts so labels do not overlap
plt.tight_layout()
plt.show()

# As distribution shows all values are left skewd, we can replace NaN with median values
for col in nan_cols:
    doge_df.fillna({col: doge_df[col].median()}, inplace=True)


# check outlier
fig,axes = plt.subplots(2,3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(nan_cols) :
    axes[i].boxplot(doge_df[col])

    # Customize individual subplot details
    axes[i].set_title(f'Boxplot of {col}', fontsize=12)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, linestyle='--', alpha=0.5)

# Adjust spacing between charts so labels do not overlap
plt.tight_layout()
plt.show()

# 1. Transform raw prices into stationary indicators
# Daily Price Spread (High vs Low)
doge_df['Price_Spread'] = (doge_df['High'] - doge_df['Low']) / doge_df['Close']

# Close vs Open return
doge_df['Intraday_Return'] = (doge_df['Close'] - doge_df['Open']) / doge_df['Open']

# 5-Day Moving Average vs Current Price
doge_df['MA5_Ratio'] = doge_df['Close'] / doge_df['Close'].rolling(window=5).mean()

# Volume Rate of Change
doge_df['Volume_Change'] = doge_df['Volume'].pct_change()
doge_df = doge_df.dropna()

# Predict 'Close' values using Random forest algorithm
doge_df['Target_Return'] = doge_df['Close'].pct_change().shift(-1)

# 3. Drop all rows where the target is missing (removes row 0 and the final row)
doge_df = doge_df.dropna(subset=['Target_Return'])
X = doge_df[['Price_Spread', 'Intraday_Return', 'MA5_Ratio', 'Volume_Change']]
y = doge_df['Target_Return']
# Split data
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

split_index = int(len(X) * 0.8)

# Slice chronologically without shuffling
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Random forest model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Important features: ", rf.feature_importances_)
# Metrics
from sklearn.metrics import r2_score, mean_squared_error
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)

print(f"New MSE: {mean_squared_error(y_test, y_pred):.6f}")
print(f"New R2 Score: {r2_score(y_test, y_pred):.4f}")

# Calculate the prediction errors
residuals = y_test - y_pred

# Plot the distribution of errors
plt.hist(residuals, bins=30, color='crimson', edgecolor='black', alpha=0.7)

# Add a reference line at 0 (where perfect predictions would sit)
plt.axvline(x=0, color='darkblue', linestyle='--', linewidth=2)

plt.title('Distribution of Model Residuals')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Count')
plt.show()