import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.arange(1, 11).reshape(-1, 1)
y = X.ravel() ** 2 + np.random.normal(0, 2, 10)

for degree in [1, 2, 5]:
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    pred = model.predict(X_poly)

    print("Degree:", degree)
    print("MSE:", round(mean_squared_error(y, pred), 2))
