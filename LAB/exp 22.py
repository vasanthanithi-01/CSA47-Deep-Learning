
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize_scalar
X, y = load_iris(return_X_y=True)
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(max_iter=5000, random_state=42,
                  early_stopping=True)
)
params = {
    'mlpclassifier__hidden_layer_sizes': [(5,), (10,), (20,)],
    'mlpclassifier__learning_rate_init': [0.001, 0.01]
}
grid = GridSearchCV(model, params, cv=3)
grid.fit(X, y)
random = RandomizedSearchCV(
    model, params, n_iter=4, cv=3, random_state=42
)
random.fit(X, y)
def objective(x):
    return (x - 10) ** 2
bayes = minimize_scalar(
    objective, bounds=(5, 20), method='bounded'
)
print("Grid Search:", grid.best_params_)
print("Random Search:", random.best_params_)
print("Bayesian Optimization:", round(bayes.x, 2))
