import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# Generate customer purchasing data
np.random.seed(42)

data = {
    "Online_Purchases": np.random.randint(1, 20, 100),
    "Store_Purchases": np.random.randint(1, 20, 100),
    "Clothing_Spending": np.random.randint(500, 5000, 100),
    "Electronics_Spending": np.random.randint(500, 5000, 100),
    "Monthly_Visits": np.random.randint(1, 30, 100)
}

df = pd.DataFrame(data)

print("Customer Purchasing Data:")
print(df.head())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(X_scaled)

# Display latent factors
factor_df = pd.DataFrame(
    factors,
    columns=["Purchase Behaviour", "Shopping Frequency"]
)

print("\nExtracted Customer Behaviour Factors:")
print(factor_df.head())

# Display factor loadings
print("\nFactor Loadings:")
print(pd.DataFrame(
    fa.components_,
    columns=df.columns,
    index=["Factor 1", "Factor 2"]
))