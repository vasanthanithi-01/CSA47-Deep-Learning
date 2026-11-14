import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# Generate student academic data
np.random.seed(42)

data = {
    "Marks": np.random.randint(40, 100, 100),
    "Attendance": np.random.randint(50, 100, 100),
    "Assignments": np.random.randint(40, 100, 100),
    "Participation": np.random.randint(30, 100, 100)
}

df = pd.DataFrame(data)

print("Student Academic Data:")
print(df.head())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(X_scaled)

# Display extracted factors
factor_df = pd.DataFrame(
    factors,
    columns=["Academic Performance", "Student Engagement"]
)

print("\nExtracted Latent Factors:")
print(factor_df.head())

print("\nFactor Loadings:")
print(pd.DataFrame(
    fa.components_,
    columns=df.columns,
    index=["Factor 1", "Factor 2"]
))